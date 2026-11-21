import audioop
#!/usr/bin/env python3
"""
Haven <-> Discord Two-Way Voice Bridge
Bridges Discord Voice Channel <---> Haven WebRTC Voice Channel in real time.
Zero latency, low memory (<100MB RAM), bidirectional audio transfer.
"""

import os
import sys
import time
import json
import asyncio
import logging
import collections
import threading
from fractions import Fraction
import urllib3
import requests
import socketio
import av
import numpy as np
import discord
import discord.ext.voice_recv as voice_recv
from dotenv import load_dotenv

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    RTCConfiguration,
    RTCIceServer,
    MediaStreamTrack
)
from aiortc.sdp import candidate_from_sdp

# Suppress insecure SSL warnings for local loopback
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("HavenDiscordBridge")

import discord.opus
try:
    for lib in ["/usr/lib/x86_64-linux-gnu/libopus.so.0", "libopus.so.0", "opus"]:
        try:
            discord.opus.load_opus(lib)
            if discord.opus.is_loaded():
                logger.info(f"✅ Loaded libopus successfully from: {lib}")
                break
        except Exception:
            pass
except Exception as e:
    logger.error(f"Error loading libopus: {e}")

def enhance_opus_sdp(sdp_str, bitrate=192000):
    lines = []
    for line in sdp_str.splitlines():
        if line.startswith("a=fmtp:") and "stereo=" not in line:
            line = f"{line};stereo=1;sprop-stereo=1;maxaveragebitrate={bitrate};cbr=0"
        lines.append(line)
    return "\r\n".join(lines) + "\r\n"


# Configuration
load_dotenv("/opt/haven-discord-bridge/.env")
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "").strip()
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_VOICE_CHANNEL_ID", "0"))

HAVEN_URL = os.getenv("HAVEN_URL", "https://127.0.0.1:3000")
HAVEN_USERNAME = os.getenv("HAVEN_USERNAME", "DiscordBridge")
HAVEN_PASSWORD = os.getenv("HAVEN_PASSWORD", "changeme")
DEFAULT_HAVEN_VOICE_CODE = os.getenv("HAVEN_VOICE_CHANNEL_CODE", "f5b41ec5")

FRAME_SIZE = 960 * 2 * 2  # 3840 bytes = 20ms @ 48kHz Stereo s16
SILENT_FRAME = b"\x00" * FRAME_SIZE


# ── Audio Streams ─────────────────────────────────────────────────────────────

class HavenPeerTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, target_user_id=None):
        super().__init__()
        self.target_user_id = target_user_id
        self._pts = 0
        self._time_base = Fraction(1, 48000)
        self._next_pts_time = time.time()
        self.queue = collections.deque(maxlen=60)
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.has_audio_event = asyncio.Event()
        self.idle_silence_count = 0
        self.sent_frames = 0
        self.last_log = time.time()

    def push_pcm(self, pcm_bytes):
        if not pcm_bytes:
            return
        with self.lock:
            self.buffer.extend(pcm_bytes)
            while len(self.buffer) >= FRAME_SIZE:
                chunk = bytes(self.buffer[:FRAME_SIZE])
                del self.buffer[:FRAME_SIZE]
                self.queue.append(chunk)
            self.has_audio_event.set()

    async def recv(self):
        now = time.time()
        wait_time = self._next_pts_time - now
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        self._next_pts_time = max(now, self._next_pts_time + 0.02)

        pcm = None
        with self.lock:
            if self.queue:
                self.idle_silence_count = 0
                pcm = self.queue.popleft()
            elif self.idle_silence_count < 5:
                self.idle_silence_count += 1
                pcm = SILENT_FRAME
            else:
                self.has_audio_event.clear()

        # If idle, sleep on event (up to 200ms) to save CPU
        if pcm is None:
            try:
                await asyncio.wait_for(self.has_audio_event.wait(), timeout=0.200)
            except (asyncio.TimeoutError, Exception):
                pass
            with self.lock:
                if self.queue:
                    pcm = self.queue.popleft()
                else:
                    pcm = SILENT_FRAME

        arr = np.frombuffer(pcm, dtype=np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(arr, format="s16", layout="stereo")
        frame.sample_rate = 48000
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += 960

        if pcm != SILENT_FRAME:
            self.sent_frames += 1
            if now - self.last_log > 5.0:
                logger.info(f"🎙️ Piping Discord -> Haven peer {self.target_user_id} ({self.sent_frames} frames forwarded)")
                self.last_log = now

        return frame


# Backwards compatibility alias
HavenOutboundTrack = HavenPeerTrack


class DiscordOutboundSource(discord.AudioSource):
    def __init__(self):
        self.queue = collections.deque(maxlen=60)
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.sent_frames = 0
        self.last_log = time.time()
        self.idle_silence_count = 0
        self.has_frames = threading.Event()

    def is_opus(self) -> bool:
        return False

    def push_pcm(self, pcm_bytes):
        if not pcm_bytes:
            return
        with self.lock:
            self.buffer.extend(pcm_bytes)
            while len(self.buffer) >= FRAME_SIZE:
                chunk = bytes(self.buffer[:FRAME_SIZE])
                del self.buffer[:FRAME_SIZE]
                self.queue.append(chunk)
            self.has_frames.set()

    def read(self) -> bytes:
        with self.lock:
            if self.queue:
                self.idle_silence_count = 0
                pcm = self.queue.popleft()
            elif self.idle_silence_count < 5:
                self.idle_silence_count += 1
                pcm = SILENT_FRAME
            else:
                self.has_frames.clear()
                pcm = None

        if pcm is not None:
            self.sent_frames += 1
            now = time.time()
            if now - self.last_log > 5.0 and pcm != SILENT_FRAME:
                logger.info(f"🔊 Piping Haven -> Discord audio ({self.sent_frames} frames forwarded)")
                self.last_log = now
            return pcm

        # Idle: wait on event (up to 200ms) without burning CPU!
        self.has_frames.wait(0.200)
        with self.lock:
            if self.queue:
                return self.queue.popleft()
        return SILENT_FRAME


class DiscordVoiceSink(voice_recv.AudioSink):
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.sent_frames = 0
        self.last_log = time.time()
        self.opus_decoder = discord.opus.Decoder()

    def cleanup(self):
        pass

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data: voice_recv.VoiceData):
        pcm = data.pcm if (data and data.pcm) else None
        # Robust fallback: if pcm is empty but raw opus packet is present, decode directly!
        if (not pcm) and data and data.opus:
            try:
                pcm = self.opus_decoder.decode(data.opus, fec=False)
            except Exception:
                pass

        if pcm:
            if len(pcm) >= 100:
                try:
                    if audioop.max(pcm, 2) < 300:
                        return
                except Exception:
                    pass
            self.sent_frames += 1
            now = time.time()
            if now - self.last_log > 3.0:
                user_name = getattr(user, "name", str(user)) if user else "Unknown"
                logger.info(f"🎙️ Received voice from Discord user {user_name} ({len(pcm)} bytes PCM, {self.sent_frames} frames -> Haven)")
                self.last_log = now
            with self.bridge.peer_tracks_lock:
                tracks = list(self.bridge.peer_tracks.values())
            for track in tracks:
                track.push_pcm(pcm)


class HavenAudioMixer:
    def __init__(self, discord_source):
        self.discord_source = discord_source
        self.user_buffers = {}  # user_id -> collections.deque(maxlen=10)
        self.lock = threading.Lock()
        self.has_audio_event = asyncio.Event()
        self.running = False
        self._task = None

    def add_user_frame(self, user_id, pcm_bytes: bytes):
        if not pcm_bytes or pcm_bytes == SILENT_FRAME:
            return
        if len(pcm_bytes) >= 100:
            try:
                if audioop.max(pcm_bytes, 2) < 300:
                    return
            except Exception:
                pass

        with self.lock:
            if user_id not in self.user_buffers:
                self.user_buffers[user_id] = collections.deque(maxlen=10)
            if len(pcm_bytes) == FRAME_SIZE:
                self.user_buffers[user_id].append(pcm_bytes)
            else:
                for i in range(0, len(pcm_bytes) - (len(pcm_bytes) % FRAME_SIZE), FRAME_SIZE):
                    chunk = pcm_bytes[i:i + FRAME_SIZE]
                    if chunk != SILENT_FRAME:
                        self.user_buffers[user_id].append(chunk)
            self.has_audio_event.set()

    def remove_user(self, user_id):
        with self.lock:
            self.user_buffers.pop(user_id, None)

    def start(self):
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._mix_loop())

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _mix_loop(self):
        next_time = time.time()
        while self.running:
            now = time.time()
            wait = next_time - now
            if wait > 0:
                await asyncio.sleep(wait)
            next_time = max(now, next_time + 0.02)

            frames_to_mix = []
            with self.lock:
                has_any = False
                for uid, q in list(self.user_buffers.items()):
                    if q:
                        frames_to_mix.append(q.popleft())
                    if len(q) > 0:
                        has_any = True
                if not has_any:
                    self.has_audio_event.clear()

            if not frames_to_mix:
                try:
                    await asyncio.wait_for(self.has_audio_event.wait(), timeout=0.200)
                except (asyncio.TimeoutError, Exception):
                    pass
                next_time = time.time()
                continue

            if len(frames_to_mix) == 1:
                mixed_pcm = frames_to_mix[0]
            else:
                samples = [np.frombuffer(f, dtype=np.int16).astype(np.float32) for f in frames_to_mix]
                sum_samples = sum(samples)
                peak = np.max(np.abs(sum_samples))
                if peak > 32767:
                    sum_samples = sum_samples * (32767.0 / peak)
                mixed_pcm = sum_samples.astype(np.int16).tobytes()

            self.discord_source.push_pcm(mixed_pcm)


# ── Bridge Orchestrator ───────────────────────────────────────────────────────

class HavenDiscordBridge:
    def __init__(self):
        self.peer_tracks = {}  # target_user_id -> HavenPeerTrack
        self.peer_tracks_lock = threading.Lock()
        self.haven_incoming_tasks = {}  # target_user_id -> asyncio.Task

        self.discord_source = DiscordOutboundSource()
        self.audio_mixer = HavenAudioMixer(self.discord_source)
        self.discord_sink = DiscordVoiceSink(self)

        intents = discord.Intents.default()
        intents.voice_states = True
        intents.guilds = True
        self.discord_client = discord.Client(intents=intents)
        self.discord_vc = None

        self.sio = None
        self.haven_token = None
        self.haven_user_id = None
        self.rtc_config = None
        self.peer_connections = {}  # userId -> RTCPeerConnection
        self.current_haven_voice_code = DEFAULT_HAVEN_VOICE_CODE
        self.user_voice_locations = {}

    def login_haven(self):
        url = f"{HAVEN_URL}/api/auth/login"
        logger.info(f"Logging in to Haven as {HAVEN_USERNAME}...")
        res = requests.post(
            url,
            json={
                "username": HAVEN_USERNAME,
                "password": HAVEN_PASSWORD,
                "eulaVersion": "2.0",
                "ageVerified": True
            },
            verify=False,
            timeout=10
        )
        if res.status_code != 200:
            raise RuntimeError(f"Haven login failed: {res.status_code} - {res.text}")
        data = res.json()
        self.haven_token = data["token"]
        self.haven_user_id = data["user"]["id"]
        logger.info(f"Haven login successful! User ID: {self.haven_user_id}")

    def fetch_ice_servers(self):
        url = f"{HAVEN_URL}/api/ice-servers"
        res = requests.get(
            url,
            headers={"Authorization": f"Bearer {self.haven_token}"},
            verify=False,
            timeout=10
        )
        if res.status_code == 200:
            servers = []
            for s in res.json().get("iceServers", []):
                urls = s.get("urls")
                if urls:
                    servers.append(RTCIceServer(urls=urls, username=s.get("username"), credential=s.get("credential")))
            self.rtc_config = RTCConfiguration(iceServers=servers)
            logger.info(f"Loaded {len(servers)} ICE server(s) for Haven.")
        else:
            self.rtc_config = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:turn.khanhs.io.vn:3478")])

    async def _handle_haven_incoming_audio(self, remote_track, user_id):
        logger.info(f"Started receiving audio from Haven user ID {user_id}")
        try:
            while True:
                frame = await remote_track.recv()
                # Use to_ndarray().tobytes() to ensure exact 3840 bytes (20ms @ 48kHz stereo s16le)
                # bytes(frame.planes[0]) returns PyAV's full 23040-byte internal buffer
                pcm = frame.to_ndarray().tobytes()
                if getattr(self, '_probe_count', 0) < 5:
                    self._probe_count = getattr(self, '_probe_count', 0) + 1
                    logger.info(f"[PROBE user {user_id}] fmt={frame.format.name} layout={frame.layout.name} rate={frame.sample_rate} samples={frame.samples} pcm_len={len(pcm)}")
                self.audio_mixer.add_user_frame(user_id, pcm)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.info(f"Ended audio stream from Haven user {user_id}: {type(e).__name__}")
        finally:
            self.audio_mixer.remove_user(user_id)

    def _setup_peer_events(self, pc, target_user_id):
        pc_tracked_ids = set()

        @pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                if track.id in pc_tracked_ids:
                    logger.warning(f"Ignoring duplicate audio track {track.id} from user {target_user_id}")
                    return
                pc_tracked_ids.add(track.id)

                # Guard against duplicate incoming audio tasks for same user
                old_task = self.haven_incoming_tasks.get(target_user_id)
                if old_task and not old_task.done():
                    logger.info(f"Replacing existing incoming audio task for user {target_user_id}")
                    old_task.cancel()

                self.haven_incoming_tasks[target_user_id] = asyncio.create_task(
                    self._handle_haven_incoming_audio(track, target_user_id)
                )

        @pc.on("connectionstatechange")
        def on_state():
            logger.info(f"Haven Peer {target_user_id} state: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                self.peer_connections.pop(target_user_id, None)
                with self.peer_tracks_lock:
                    self.peer_tracks.pop(target_user_id, None)
                task = self.haven_incoming_tasks.pop(target_user_id, None)
                if task and not task.done():
                    task.cancel()
                self.audio_mixer.remove_user(target_user_id)

    async def _create_haven_peer_and_offer(self, target_user_id):
        if target_user_id == self.haven_user_id:
            return

        old_task = self.haven_incoming_tasks.pop(target_user_id, None)
        if old_task and not old_task.done():
            old_task.cancel()
        with self.peer_tracks_lock:
            self.peer_tracks.pop(target_user_id, None)
        self.audio_mixer.remove_user(target_user_id)

        old_pc = self.peer_connections.pop(target_user_id, None)
        if old_pc:
            async def _close_bg2(p):
                try:
                    await asyncio.wait_for(p.close(), timeout=1.5)
                except Exception:
                    pass
            asyncio.create_task(_close_bg2(old_pc))

        pc = RTCPeerConnection(configuration=self.rtc_config)
        self.peer_connections[target_user_id] = pc

        # Create independent peer track
        peer_track = HavenPeerTrack(target_user_id=target_user_id)
        with self.peer_tracks_lock:
            self.peer_tracks[target_user_id] = peer_track
        pc.addTrack(peer_track)

        self._setup_peer_events(pc, target_user_id)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        await self.sio.emit("voice-offer", {
            "code": self.current_haven_voice_code,
            "targetUserId": target_user_id,
            "offer": {
                "type": pc.localDescription.type,
                "sdp": enhance_opus_sdp(pc.localDescription.sdp, 192000)
            }
        })

    async def join_haven_voice(self, code):
        if self.current_haven_voice_code and self.current_haven_voice_code != code:
            logger.info(f"Leaving Haven voice room {self.current_haven_voice_code}...")
            await self.sio.emit("voice-leave", {"code": self.current_haven_voice_code})
            for uid, pc in list(self.peer_connections.items()):
                try:
                    await pc.close()
                except Exception:
                    pass
            self.peer_connections.clear()
            with self.peer_tracks_lock:
                self.peer_tracks.clear()
            for uid, task in list(self.haven_incoming_tasks.items()):
                if not task.done():
                    task.cancel()
            self.haven_incoming_tasks.clear()
            with self.audio_mixer.lock:
                self.audio_mixer.user_buffers.clear()

        self.current_haven_voice_code = code
        logger.info(f"Joining Haven voice room {code}...")
        await self.sio.emit("voice-join", {"code": code})
        await self.sio.emit("voice-mute-state", {"code": code, "muted": False})
        await self.sio.emit("voice-speaking", {"speaking": True, "code": code})

    async def _ensure_discord_voice_connected(self):
        try:
            if not self.discord_client.is_ready():
                return
            channel = self.discord_client.get_channel(DISCORD_CHANNEL_ID)
            if not channel:
                channel = await self.discord_client.fetch_channel(DISCORD_CHANNEL_ID)

            if self.discord_vc and self.discord_vc.is_connected():
                if getattr(self.discord_vc.channel, "id", 0) == DISCORD_CHANNEL_ID:
                    return
                try:
                    await self.discord_vc.disconnect(force=True)
                except Exception:
                    pass

            logger.info(f"Connecting to Discord Voice Channel: {channel.name} (Guild: {channel.guild.name})...")
            self.discord_vc = await channel.connect(cls=voice_recv.VoiceRecvClient, reconnect=True, timeout=15.0)
            self.discord_vc.play(self.discord_source)
            self.discord_vc.listen(self.discord_sink)
            logger.info("✅ Discord Voice connected & 2-way audio loop started!")
        except Exception as e:
            logger.error(f"Failed to connect to Discord voice: {e}")

    def _setup_discord(self):
        @self.discord_client.event
        async def on_ready():
            logger.info(f"Discord Bot online: {self.discord_client.user.name} ({self.discord_client.user.id})")
            await self._ensure_discord_voice_connected()

        @self.discord_client.event
        async def on_voice_state_update(member, before, after):
            if member.id == self.discord_client.user.id:
                if after.channel is None or after.channel.id != DISCORD_CHANNEL_ID:
                    logger.warning("Bot voice state changed unexpectedly! Rejoining target Discord voice channel...")
                    await asyncio.sleep(2)
                    await self._ensure_discord_voice_connected()

    def _setup_haven(self):
        self.login_haven()
        self.fetch_ice_servers()

        self.sio = socketio.AsyncClient(ssl_verify=False, reconnection=True)

        @self.sio.event
        async def connect():
            logger.info("Haven Socket.IO connected! Initializing channels...")
            await self.sio.emit("get-channels")
            await self.join_haven_voice(self.current_haven_voice_code)

        @self.sio.on("voice-users-update")
        async def on_voice_update(data):
            code = data.get("code")
            users = data.get("users", [])
            for u in users:
                uid = u.get("id")
                uname = u.get("username")
                if uid and uid != self.haven_user_id:
                    self.user_voice_locations[uid] = code
                    if uname:
                        self.user_voice_locations[uname.lower()] = code

        @self.sio.on("voice-existing-users")
        async def on_existing_users(data):
            code = data.get("channelCode")
            users = data.get("users", [])
            logger.info(f"Haven voice-existing-users ({code}): {len(users)} users")
            for u in users:
                uid = u.get("id")
                if uid and uid != self.haven_user_id:
                    await self._create_haven_peer_and_offer(uid)

        @self.sio.on("voice-user-joined")
        async def on_user_joined(data):
            code = data.get("channelCode")
            user = data.get("user", {})
            uid = user.get("id")
            uname = user.get("username")
            if uid and uid != self.haven_user_id:
                logger.info(f"User joined Haven voice {code}: {uname} (ID: {uid})")
                self.user_voice_locations[uid] = code
                if uname:
                    self.user_voice_locations[uname.lower()] = code
                if self.current_haven_voice_code == code:
                    logger.info(f"Proactively offering bridge audio track to newcomer {uname} ({uid})")
                    await self._create_haven_peer_and_offer(uid)

        @self.sio.on("voice-offer")
        async def on_voice_offer(data):
            from_user = data.get("from", {})
            target_user_id = from_user.get("id")
            offer_data = data.get("offer", {})
            if not target_user_id or target_user_id == self.haven_user_id:
                return

            logger.info(f"Received Haven voice-offer from {from_user.get('username')} ({target_user_id})")
            old_task = self.haven_incoming_tasks.pop(target_user_id, None)
            if old_task and not old_task.done():
                old_task.cancel()
            with self.peer_tracks_lock:
                self.peer_tracks.pop(target_user_id, None)
            self.audio_mixer.remove_user(target_user_id)

            old_pc = self.peer_connections.pop(target_user_id, None)
            if old_pc:
                async def _close_bg(p):
                    try:
                        await asyncio.wait_for(p.close(), timeout=1.5)
                    except Exception:
                        pass
                asyncio.create_task(_close_bg(old_pc))

            pc = RTCPeerConnection(configuration=self.rtc_config)
            self.peer_connections[target_user_id] = pc

            peer_track = HavenPeerTrack(target_user_id=target_user_id)
            with self.peer_tracks_lock:
                self.peer_tracks[target_user_id] = peer_track
            pc.addTrack(peer_track)

            self._setup_peer_events(pc, target_user_id)

            offer = RTCSessionDescription(sdp=offer_data["sdp"], type=offer_data["type"])
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            await self.sio.emit("voice-answer", {
                "code": self.current_haven_voice_code,
                "targetUserId": target_user_id,
                "answer": {"type": pc.localDescription.type, "sdp": enhance_opus_sdp(pc.localDescription.sdp, 192000)}
            })

        @self.sio.on("voice-answer")
        async def on_voice_answer(data):
            target_user_id = data.get("from", {}).get("id")
            answer_data = data.get("answer", {})
            if target_user_id in self.peer_connections:
                pc = self.peer_connections[target_user_id]
                answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
                await pc.setRemoteDescription(answer)

        @self.sio.on("voice-ice-candidate")
        async def on_ice(data):
            target_user_id = data.get("from", {}).get("id")
            cand_data = data.get("candidate")
            if target_user_id in self.peer_connections and cand_data:
                pc = self.peer_connections[target_user_id]
                cand_str = cand_data.get("candidate") if isinstance(cand_data, dict) else str(cand_data)
                if cand_str:
                    try:
                        c = candidate_from_sdp(cand_str)
                        c.sdpMid = cand_data.get("sdpMid") if isinstance(cand_data, dict) else None
                        c.sdpMLineIndex = cand_data.get("sdpMLineIndex") if isinstance(cand_data, dict) else None
                        await pc.addIceCandidate(c)
                    except Exception:
                        pass

        @self.sio.on("voice-user-left")
        async def on_user_left(data):
            uid = data.get("user", {}).get("id")
            old_pc = self.peer_connections.pop(uid, None)
            if old_pc:
                try:
                    await old_pc.close()
                except Exception:
                    pass
            with self.peer_tracks_lock:
                self.peer_tracks.pop(uid, None)
            task = self.haven_incoming_tasks.pop(uid, None)
            if task and not task.done():
                task.cancel()
            self.audio_mixer.remove_user(uid)

        # Chat commands: "!bridge here"
        @self.sio.on("message")
        @self.sio.on("new-message")
        async def on_chat_message(data):
            msg = data.get("message", data)
            content = (msg.get("content") or "").strip()
            sender = msg.get("user") or {}
            sender_id = sender.get("id")
            sender_name = sender.get("username", "")

            if sender_id == self.haven_user_id:
                return

            if content.lower() in ["!bridge here", "!bridge"]:
                user_loc = self.user_voice_locations.get(sender_id) or self.user_voice_locations.get(sender_name.lower())
                if user_loc:
                    await self.join_haven_voice(user_loc)
                    await self.sio.emit("send-message", {
                        "code": user_loc,
                        "content": f"🎙️ **Discord Bridge** đã chuyển đến kênh voice `{user_loc}` theo yêu cầu của **{sender_name}**!"
                    })

    async def _voice_watchdog(self):
        """Keeps both Haven and Discord voice connections alive 24/7."""
        logger.info("🛡️ 24/7 Voice Watchdog started (checks every 15s).")
        while True:
            await asyncio.sleep(15)
            # 1. Maintain Discord Voice
            try:
                if self.discord_client.is_ready():
                    if not self.discord_vc or not self.discord_vc.is_connected():
                        logger.warning("Watchdog detected Discord voice disconnected! Reconnecting...")
                        await self._ensure_discord_voice_connected()
            except Exception as e:
                logger.error(f"Watchdog Discord check error: {e}")

            # 2. Maintain Haven Voice (do NOT re-emit voice-join if already connected to prevent kick/rejoin loops)
            try:
                if self.sio and not self.sio.connected:
                    logger.warning("Watchdog detected Haven Socket.IO disconnected! Reconnecting...")
                    try:
                        await self.sio.connect(HAVEN_URL, auth={"token": self.haven_token}, transports=["websocket"])
                    except Exception as ce:
                        logger.error(f"Watchdog Haven reconnect error: {ce}")
            except Exception as e:
                logger.error(f"Watchdog Haven check error: {e}")

    async def run(self):
        self._setup_haven()
        self._setup_discord()
        self.audio_mixer.start()
        asyncio.create_task(self._voice_watchdog())

        def custom_exception_handler(loop, context):
            msg = context.get("message", "")
            exc = context.get("exception")
            if exc and "sendto" in str(exc) and "NoneType" in str(exc):
                return
            if "Transaction.__retry" in msg:
                return
            if exc and "InvalidStateError" in str(type(exc)):
                return
            loop.default_exception_handler(context)

        loop = asyncio.get_running_loop()
        loop.set_exception_handler(custom_exception_handler)

        await asyncio.gather(
            self.sio.connect(HAVEN_URL, auth={"token": self.haven_token}, transports=["websocket"]),
            self.discord_client.start(DISCORD_TOKEN)
        )


if __name__ == "__main__":
    bridge = HavenDiscordBridge()
    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        pass
