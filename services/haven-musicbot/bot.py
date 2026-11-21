#!/usr/bin/env python3
"""
Haven Music Bot (Optimized WebRTC Audio Broadcaster)
High-fidelity, crackle-free audio streamer for Haven Chat (ghcr.io/ancsemi/haven).
Features:
- Master ffmpeg producer with per-peer frame fan-out (eliminates audio interleaving & crackling)
- Full YouTube & SoundCloud playlist support (instant --flat-playlist extraction + lazy stream resolving)
- Non-blocking WebRTC connection teardown (prevents event loop stalls)
"""

import os
import sys
import time
import json
import asyncio
import logging
import subprocess
import collections
import threading
from fractions import Fraction
import urllib3
import requests
import socketio
import av
import numpy as np
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
logger = logging.getLogger("HavenMusicBot")

# Load environment configuration
load_dotenv("/opt/haven-musicbot/.env")
HAVEN_URL = os.getenv("HAVEN_URL", "https://127.0.0.1:3000")
HAVEN_PUBLIC_URL = os.getenv("HAVEN_PUBLIC_URL", "https://haven.khanhs.io.vn")
BOT_USERNAME = os.getenv("BOT_USERNAME", "MusicBot")
BOT_PASSWORD = os.getenv("BOT_PASSWORD", "changeme")
TARGET_CHANNEL_CODE = os.getenv("TARGET_CHANNEL_CODE", "f5b41ec5")
DEFAULT_VOLUME = int(os.getenv("DEFAULT_VOLUME", "75"))
IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_SECONDS", "300"))

FRAME_SAMPLES = 960  # 20ms @ 48kHz
FRAME_BYTES = FRAME_SAMPLES * 2 * 2  # 3840 bytes (stereo s16le)
SILENT_PCM = b"\x00" * FRAME_BYTES


class BroadcasterPeerTrack(MediaStreamTrack):
    """
    Dedicated MediaStreamTrack for each connected WebRTC peer.
    Pulls synchronized frames from the master AudioBroadcaster.
    """
    kind = "audio"

    def __init__(self, broadcaster, user_id):
        super().__init__()
        self.broadcaster = broadcaster
        self.user_id = user_id
        self._pts = 0
        self._time_base = Fraction(1, 48000)

    async def recv(self):
        raw_pcm = await self.broadcaster.get_frame_for_peer(self.user_id)
        if not raw_pcm or len(raw_pcm) < FRAME_BYTES:
            raw_pcm = SILENT_PCM

        arr = np.frombuffer(raw_pcm, dtype=np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(arr, format="s16", layout="stereo")
        frame.sample_rate = 48000
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += FRAME_SAMPLES

        return frame


class AudioBroadcaster:
    """
    Master audio stream producer that runs ffmpeg once and fans out
    the EXACT same 48kHz stereo PCM frames to all active peer tracks.
    """
    def __init__(self, on_finish_callback=None):
        self.on_finish_callback = on_finish_callback
        self.process = None
        self.producer_task = None
        self.is_playing = False
        self.current_track = None
        self.volume = DEFAULT_VOLUME
        self.peer_queues = {}   # user_id -> deque(maxlen=20)
        self.peer_events = {}   # user_id -> asyncio.Event()
        self.lock = threading.Lock()

    def create_track_for_peer(self, user_id):
        with self.lock:
            self.peer_queues[user_id] = collections.deque(maxlen=15)
            self.peer_events[user_id] = asyncio.Event()
        logger.info(f"Registered audio broadcast track for peer {user_id}")
        return BroadcasterPeerTrack(self, user_id)

    def remove_peer(self, user_id):
        with self.lock:
            self.peer_queues.pop(user_id, None)
            ev = self.peer_events.pop(user_id, None)
            if ev:
                ev.set()
        logger.info(f"Removed audio broadcast track for peer {user_id}")

    async def get_frame_for_peer(self, user_id):
        with self.lock:
            q = self.peer_queues.get(user_id)
            ev = self.peer_events.get(user_id)

        if q is None:
            return SILENT_PCM

        with self.lock:
            if len(q) > 0:
                return q.popleft()
            if ev:
                ev.clear()

        if ev:
            try:
                await asyncio.wait_for(ev.wait(), timeout=0.020)
            except (asyncio.TimeoutError, Exception):
                pass

        with self.lock:
            if q and len(q) > 0:
                return q.popleft()

        return SILENT_PCM

    async def start_stream(self, track_info):
        self.stop_stream()
        self.current_track = track_info
        self.is_playing = True

        # Clear peer queues
        with self.lock:
            for q in self.peer_queues.values():
                q.clear()

        url = track_info.get("webpage_url") or track_info.get("stream_url")
        logger.info(f"Starting master ffmpeg audio stream for: {track_info['title']} ({url})")

        shell_cmd = (
            f'/opt/haven-musicbot/venv/bin/yt-dlp -o - -f "bestaudio/best" --no-playlist '
            f'--extractor-args "youtube:player_client=android,web" "{url}" '
            f'| ffmpeg -nostdin -i pipe:0 -vn -f s16le -ac 2 -ar 48000 -loglevel error pipe:1'
        )

        self.process = await asyncio.create_subprocess_shell(
            shell_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )

        self.producer_task = asyncio.create_task(self._producer_loop())

    def stop_stream(self):
        self.is_playing = False
        self.current_track = None
        if self.producer_task and not self.producer_task.done():
            self.producer_task.cancel()
            self.producer_task = None
        if self.process:
            try:
                self.process.terminate()
            except Exception:
                pass
            self.process = None

    async def _producer_loop(self):
        next_pts_time = time.time()
        try:
            while self.is_playing and self.process and self.process.stdout:
                now = time.time()
                wait_time = next_pts_time - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                next_pts_time = max(now, next_pts_time + 0.02)

                try:
                    raw_bytes = await self.process.stdout.readexactly(FRAME_BYTES)
                except (asyncio.IncompleteReadError, Exception):
                    logger.info("Audio stream reached EOF or pipe ended.")
                    break

                if not raw_bytes or len(raw_bytes) < FRAME_BYTES:
                    break

                # Apply volume scaling if needed
                if self.volume != 100:
                    samples = np.frombuffer(raw_bytes, dtype=np.int16)
                    scaled = np.clip(samples * (self.volume / 100.0), -32768, 32767).astype(np.int16)
                    raw_bytes = scaled.tobytes()

                # Fan-out exact same PCM frame to all peers
                with self.lock:
                    for q in self.peer_queues.values():
                        q.append(raw_bytes)
                    for ev in self.peer_events.values():
                        ev.set()
        except asyncio.CancelledError:
            pass
        finally:
            self.stop_stream()
            if self.on_finish_callback:
                asyncio.create_task(self.on_finish_callback())


class YtDlpExtractor:
    """Extracts streaming audio URLs or full playlists from YouTube / SoundCloud."""

    YTDLP_BIN = "/opt/haven-musicbot/venv/bin/yt-dlp"

    @classmethod
    def extract_info(cls, query):
        clean_query = query.strip()
        is_direct_url = clean_query.startswith("http://") or clean_query.startswith("https://")
        is_playlist = ("playlist?list=" in clean_query or "&list=" in clean_query or "open.spotify.com/playlist" in clean_query)

        # ── Fast Playlist Extraction ───────────────────────────
        if is_playlist:
            cmd = [
                cls.YTDLP_BIN,
                "--dump-single-json",
                "--flat-playlist",
                "--skip-download",
                "--extractor-args", "youtube:player_client=android,web",
                clean_query
            ]
            try:
                logger.info(f"Extracting flat playlist: {clean_query}")
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
                if res.returncode == 0 and res.stdout.strip():
                    data = json.loads(res.stdout)
                    entries = data.get("entries") or []
                    if entries:
                        playlist_title = data.get("title") or "YouTube Playlist"
                        tracks = []
                        for entry in entries:
                            vid = entry.get("id")
                            vtitle = entry.get("title") or "Unknown Title"
                            if not vid:
                                continue
                            vurl = f"https://www.youtube.com/watch?v={vid}"
                            dur = int(entry.get("duration") or 0)
                            tracks.append({
                                "title": vtitle,
                                "webpage_url": vurl,
                                "stream_url": None,  # Lazy extraction on playback
                                "duration": dur,
                                "uploader": entry.get("uploader", "Unknown Uploader")
                            })
                        return {
                            "is_playlist": True,
                            "title": playlist_title,
                            "tracks": tracks
                        }
            except Exception as e:
                logger.error(f"Flat playlist extraction failed: {e}")

        # ── Single Video / Search Query ────────────────────────
        search_query = clean_query
        if not is_direct_url:
            if clean_query.lower().startswith("sc:") or clean_query.lower().startswith("soundcloud:"):
                clean = clean_query.split(":", 1)[1].strip()
                search_query = f"scsearch1:{clean}"
            else:
                search_query = f"ytsearch1:{clean_query}"

        cmd = [
            cls.YTDLP_BIN,
            "--dump-single-json",
            "-f", "bestaudio/best",
            "--no-playlist",
            "--skip-download",
            "--extractor-args", "youtube:player_client=android,web",
            search_query
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            if res.returncode != 0:
                logger.error(f"yt-dlp extraction error: {res.stderr}")
                return None

            data = json.loads(res.stdout)
            if "entries" in data and len(data["entries"]) > 0:
                data = data["entries"][0]

            stream_url = data.get("url")
            if not stream_url:
                return None

            return {
                "is_playlist": False,
                "title": data.get("title", "Unknown Title"),
                "stream_url": stream_url,
                "duration": int(data.get("duration") or 0),
                "webpage_url": data.get("webpage_url", clean_query),
                "uploader": data.get("uploader", "Unknown Uploader")
            }
        except Exception as e:
            logger.error(f"Failed to extract audio with yt-dlp: {e}")
            return None

    @classmethod
    def resolve_single_stream(cls, webpage_url):
        """Lazily extracts stream info for a single track from playlist."""
        cmd = [
            cls.YTDLP_BIN,
            "--dump-single-json",
            "-f", "bestaudio/best",
            "--no-playlist",
            "--skip-download",
            "--extractor-args", "youtube:player_client=android,web",
            webpage_url
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                if "entries" in data and len(data["entries"]) > 0:
                    data = data["entries"][0]
                return {
                    "title": data.get("title", "Unknown Title"),
                    "stream_url": data.get("url"),
                    "duration": int(data.get("duration") or 0),
                    "webpage_url": data.get("webpage_url", webpage_url),
                    "uploader": data.get("uploader", "Unknown Uploader")
                }
        except Exception as e:
            logger.error(f"Lazy stream resolve error for {webpage_url}: {e}")
        return None


class HavenMusicBot:
    def __init__(self):
        self.sio = None
        self.jwt_token = None
        self.user_id = None
        self.rtc_config = None
        self.broadcaster = AudioBroadcaster(on_finish_callback=self._on_track_finished)
        self.queue = []
        self.peer_connections = {}  # targetUserId -> RTCPeerConnection
        self.in_voice = False
        self.current_voice_code = None
        self.active_listeners = set()
        self.idle_task = None
        self.user_voice_locations = {}
        self.channel_names = {}

    def login(self):
        url = f"{HAVEN_URL}/api/auth/login"
        logger.info(f"Authenticating as {BOT_USERNAME} at {url}...")
        res = requests.post(
            url,
            json={
                "username": BOT_USERNAME,
                "password": BOT_PASSWORD,
                "eulaVersion": "2.0",
                "ageVerified": True
            },
            verify=False,
            timeout=10
        )
        if res.status_code != 200:
            raise RuntimeError(f"Login failed: {res.text}")
        data = res.json()
        self.jwt_token = data.get("token")
        self.user_id = data.get("user", {}).get("id")
        logger.info(f"Logged in successfully! (User ID: {self.user_id})")

    def fetch_ice_servers(self):
        url = f"{HAVEN_URL}/api/ice-servers"
        ice_servers = []
        try:
            res = requests.get(
                url,
                headers={"Authorization": f"Bearer {self.jwt_token}"},
                verify=False,
                timeout=10
            )
            if res.status_code == 200:
                rtc_data = res.json()
                for s in rtc_data.get("iceServers", []):
                    urls = s.get("urls")
                    if isinstance(urls, str):
                        urls = [urls]
                    if urls:
                        ice_servers.append(
                            RTCIceServer(
                                urls=urls,
                                username=s.get("username"),
                                credential=s.get("credential")
                            )
                        )
                logger.info(f"Fetched {len(ice_servers)} ICE servers from {url}.")
            else:
                logger.warning(f"Failed to fetch ICE servers (HTTP {res.status_code}), using Coturn defaults.")
        except Exception as e:
            logger.error(f"Error fetching ICE servers: {e}")

        if not ice_servers:
            ice_servers = [
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:turn.khanhs.io.vn:3478"]),
                RTCIceServer(
                    urls=["turn:turn.khanhs.io.vn:3478"],
                    username=f"1795366355:{BOT_USERNAME}",
                    credential="xmZ5wqn8/1mEgaGp5Wc0QUqXrZE="
                )
            ]
        self.rtc_config = RTCConfiguration(iceServers=ice_servers)
        logger.info(f"Configured WebRTC with {len(ice_servers)} ICE servers.")

    async def send_chat(self, channel_code, text):
        if not self.sio:
            return
        await self.sio.emit("send-message", {
            "channelCode": channel_code,
            "content": text
        })

    def get_user_current_voice_channel(self, author_id, author_name):
        if author_id and author_id in self.user_voice_locations:
            return self.user_voice_locations[author_id]
        if author_name and author_name.lower() in self.user_voice_locations:
            return self.user_voice_locations[author_name.lower()]
        return None

    def _reset_idle_timer(self):
        if self.idle_task and not self.idle_task.done():
            self.idle_task.cancel()
        self.idle_task = asyncio.create_task(self._idle_timeout_worker())

    async def _idle_timeout_worker(self):
        try:
            await asyncio.sleep(IDLE_TIMEOUT_SECONDS)
            if self.in_voice and not self.broadcaster.is_playing and len(self.queue) == 0:
                logger.info(f"Idle timeout reached ({IDLE_TIMEOUT_SECONDS}s). Leaving voice channel.")
                if self.current_voice_code:
                    await self.send_chat(
                        self.current_voice_code,
                        f"💤 Bot rời phòng thoại do không phát nhạc trong {IDLE_TIMEOUT_SECONDS // 60} phút."
                    )
                await self.leave_voice()
        except asyncio.CancelledError:
            pass

    async def _set_speaking(self, is_speaking):
        if self.sio and self.current_voice_code:
            await self.sio.emit("voice-speaking", {
                "code": self.current_voice_code,
                "speaking": is_speaking
            })

    async def _on_track_finished(self):
        logger.info("Current track finished playback.")
        if len(self.queue) > 0:
            next_track = self.queue.pop(0)
            await self._play_track(next_track)
        else:
            await self._set_speaking(False)
            if self.current_voice_code:
                await self.send_chat(self.current_voice_code, "✅ Đã phát hết danh sách bài hát trong hàng đợi.")
            self._reset_idle_timer()

    async def _play_track(self, track_info):
        # Resolve single streaming URL lazily if not yet resolved
        if not track_info.get("stream_url"):
            logger.info(f"Lazily resolving stream URL for: {track_info['title']}")
            res = await asyncio.to_thread(YtDlpExtractor.resolve_single_stream, track_info["webpage_url"])
            if res and res.get("stream_url"):
                track_info.update(res)
            else:
                logger.error(f"Could not resolve stream URL for {track_info['title']}")
                await self._on_track_finished()
                return

        # Ensure all listeners are connected before audio starts flowing
        await self._ensure_all_listeners_connected()

        await self.broadcaster.start_stream(track_info)
        await self._set_speaking(True)
        dur = track_info.get("duration", 0)
        dur_str = f"{dur // 60}:{dur % 60:02d}" if dur > 0 else "Live"
        msg = f"🎶 **Đang phát:** `{track_info['title']}` `[{dur_str}]`"
        if track_info.get("requester"):
            msg += f" (yêu cầu bởi *{track_info['requester']}*)"
        if self.current_voice_code:
            await self.send_chat(self.current_voice_code, msg)
        self._reset_idle_timer()

    async def _ensure_all_listeners_connected(self):
        """Ensure every active listener in current room has a healthy RTCPeerConnection."""
        if not self.in_voice or not self.current_voice_code:
            return
        for uid in list(self.active_listeners):
            if uid == self.user_id:
                continue
            pc = self.peer_connections.get(uid)
            if not pc or pc.connectionState in ["failed", "closed", "disconnected"]:
                logger.info(f"Listener {uid} WebRTC state: {getattr(pc, 'connectionState', 'None')} -> Re-offering audio track...")
                await self._create_peer_and_offer(uid)

    async def join_voice(self, channel_code):
        if self.in_voice and self.current_voice_code == channel_code:
            await self._ensure_all_listeners_connected()
            return
        if self.in_voice:
            await self.leave_voice()

        self.current_voice_code = channel_code
        self.in_voice = True
        ch_name = self.channel_names.get(channel_code, channel_code)
        logger.info(f"Joining voice channel {ch_name} ({channel_code})...")
        await self.sio.emit("voice-join", {"code": channel_code})

    async def leave_voice(self):
        if not self.in_voice:
            return
        logger.info(f"Leaving voice channel {self.current_voice_code}...")
        self.broadcaster.stop_stream()
        self.queue.clear()
        await self._set_speaking(False)

        # Close all active WebRTC peer connections safely with timeouts
        for user_id, pc in list(self.peer_connections.items()):
            self.broadcaster.remove_peer(user_id)
            try:
                await asyncio.wait_for(pc.close(), timeout=1.0)
            except Exception:
                pass
        self.peer_connections.clear()
        self.active_listeners.clear()

        if self.sio and self.current_voice_code:
            await self.sio.emit("voice-leave", {"code": self.current_voice_code})

        self.in_voice = False
        self.current_voice_code = None
        if self.idle_task:
            self.idle_task.cancel()
            self.idle_task = None

    async def _safe_close_pc(self, pc, user_id):
        self.broadcaster.remove_peer(user_id)
        if pc:
            try:
                await asyncio.wait_for(pc.close(), timeout=1.5)
            except Exception:
                pass

    async def _create_peer_and_offer(self, target_user_id):
        if target_user_id == self.user_id:
            return

        old_pc = self.peer_connections.pop(target_user_id, None)
        if old_pc:
            asyncio.create_task(self._safe_close_pc(old_pc, target_user_id))

        pc = RTCPeerConnection(configuration=self.rtc_config)
        self.peer_connections[target_user_id] = pc

        # Dedicated peer track from master broadcaster
        track = self.broadcaster.create_track_for_peer(target_user_id)
        pc.addTrack(track)

        @pc.on("connectionstatechange")
        def on_state():
            logger.info(f"Peer {target_user_id} connection state: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                asyncio.create_task(self._safe_close_pc(pc, target_user_id))
                self.peer_connections.pop(target_user_id, None)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        await self.sio.emit("voice-offer", {
            "code": self.current_voice_code,
            "targetUserId": target_user_id,
            "offer": {
                "type": pc.localDescription.type,
                "sdp": pc.localDescription.sdp
            }
        })

    async def handle_chat_command(self, channel_code, author_id, author_name, content):
        text = content.strip()
        if not text.startswith("!"):
            return

        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ["!play", "!p"]:
            if not args:
                await self.send_chat(channel_code, "❌ Vui lòng nhập link hoặc tên bài hát. Ví dụ: `!play lofi hip hop`")
                return

            user_voice_code = self.get_user_current_voice_channel(author_id, author_name)
            if not user_voice_code:
                await self.send_chat(
                    channel_code,
                    f"⚠️ **{author_name}** chưa tham gia kênh thoại nào! Hãy vào một phòng thoại rồi gõ lại lệnh nhé."
                )
                return

            target_ch_name = self.channel_names.get(user_voice_code, user_voice_code)
            if not self.in_voice or self.current_voice_code != user_voice_code:
                await self.send_chat(channel_code, f"🎯 Đang vào phòng thoại **{target_ch_name}** cùng bạn...")
                await self.join_voice(user_voice_code)
            else:
                await self._ensure_all_listeners_connected()

            await self.send_chat(channel_code, f"🔍 Đang tìm kiếm: `{args}`...")
            info = await asyncio.to_thread(YtDlpExtractor.extract_info, args)
            if not info:
                await self.send_chat(channel_code, f"❌ Không tìm thấy bài hát hoặc lỗi trích xuất: `{args}`")
                return

            # Playlist Handling
            if info.get("is_playlist"):
                tracks = info.get("tracks", [])
                if not tracks:
                    await self.send_chat(channel_code, f"❌ Playlist trống hoặc không có bài hát công khai: `{args}`")
                    return

                for t in tracks:
                    t["requester"] = author_name

                first_track = tracks[0]
                remaining_tracks = tracks[1:]

                if not self.broadcaster.is_playing:
                    await self._play_track(first_track)
                    self.queue.extend(remaining_tracks)
                    await self.send_chat(
                        channel_code,
                        f"📑 Đã tải playlist **{info['title']}** ({len(tracks)} bài hát) vào hàng đợi! Đang phát: `{first_track['title']}`"
                    )
                else:
                    self.queue.extend(tracks)
                    await self.send_chat(
                        channel_code,
                        f"📑 Đã thêm playlist **{info['title']}** ({len(tracks)} bài hát) vào hàng đợi!"
                    )
                return

            # Single Track Handling
            info["requester"] = author_name
            if not self.broadcaster.is_playing:
                await self._play_track(info)
            else:
                self.queue.append(info)
                pos = len(self.queue)
                dur = info.get("duration", 0)
                dur_str = f"{dur // 60}:{dur % 60:02d}" if dur > 0 else "Live"
                await self.send_chat(
                    channel_code,
                    f"➕ Đã thêm vào hàng đợi **#{pos}**: `{info['title']}` `[{dur_str}]`"
                )

        elif cmd in ["!skip", "!s"]:
            if not self.broadcaster.is_playing:
                await self.send_chat(channel_code, "⚠️ Hiện không có bài hát nào đang phát.")
                return
            await self.send_chat(channel_code, "⏭️ Đang chuyển bài...")
            self.broadcaster.stop_stream()
            await self._on_track_finished()

        elif cmd == "!stop":
            await self.send_chat(channel_code, "⏹️ Đã dừng phát nhạc và làm trống hàng đợi.")
            await self.leave_voice()

        elif cmd == "!join":
            user_voice_code = self.get_user_current_voice_channel(author_id, author_name)
            target_code = user_voice_code or channel_code
            target_name = self.channel_names.get(target_code, target_code)
            await self.send_chat(channel_code, f"🎙️ Đang kết nối vào phòng thoại **{target_name}**...")
            await self.join_voice(target_code)
            await self._ensure_all_listeners_connected()

        elif cmd == "!leave":
            if self.in_voice:
                await self.send_chat(channel_code, "👋 Đã rời phòng thoại.")
                await self.leave_voice()
            else:
                await self.send_chat(channel_code, "⚠️ Bot hiện không ở trong phòng thoại nào.")

        elif cmd in ["!queue", "!q"]:
            if not self.broadcaster.is_playing and len(self.queue) == 0:
                await self.send_chat(channel_code, "📭 Hàng đợi bài hát đang trống.")
                return

            lines = ["📜 **Danh sách phát hiện tại:**"]
            if self.broadcaster.current_track:
                lines.append(f"▶️ **Đang phát:** `{self.broadcaster.current_track['title']}`")
            if len(self.queue) > 0:
                for idx, t in enumerate(self.queue[:10], 1):
                    lines.append(f"{idx}. `{t['title']}` (bởi *{t.get('requester', 'ai đó')}*)")
                if len(self.queue) > 10:
                    lines.append(f"...và còn {len(self.queue) - 10} bài nữa.")
            await self.send_chat(channel_code, "\n".join(lines))

        elif cmd in ["!vol", "!volume"]:
            if not args or not args.isdigit():
                await self.send_chat(channel_code, f"🔊 Âm lượng hiện tại: `{self.broadcaster.volume}%`. Dùng: `!vol <0-100>`")
                return
            val = max(0, min(100, int(args)))
            self.broadcaster.volume = val
            await self.send_chat(channel_code, f"🔊 Đã điều chỉnh âm lượng: `{val}%`")

        elif cmd == "!np":
            if not self.broadcaster.is_playing or not self.broadcaster.current_track:
                await self.send_chat(channel_code, "⚠️ Không có bài hát nào đang phát.")
                return
            cur = self.broadcaster.current_track
            await self.send_chat(
                channel_code,
                f"🎵 **Đang phát:** `{cur['title']}`\n🔗 Link: {cur.get('webpage_url', 'N/A')}\n🔊 Âm lượng: `{self.broadcaster.volume}%`"
            )

        elif cmd == "!help":
            help_msg = (
                "🤖 **Lệnh điều khiển Music Bot:**\n"
                "• `!play <tên hoặc link>`: Tự động vào phòng thoại của bạn và phát nhạc (hỗ trợ playlist!)\n"
                "• `!skip`: Bỏ qua bài đang phát\n"
                "• `!stop`: Dừng phát và rời phòng thoại\n"
                "• `!join`: Gọi bot vào phòng thoại của bạn\n"
                "• `!leave`: Cho bot rời phòng thoại\n"
                "• `!queue`: Xem danh sách bài hát đang chờ\n"
                "• `!vol <0-100>`: Chỉnh âm lượng (mặc định 75%)\n"
                "• `!np`: Xem thông tin bài hát đang phát\n"
                "• `!help`: Hiện danh sách hướng dẫn này"
            )
            await self.send_chat(channel_code, help_msg)

    async def run(self):
        self.login()
        self.fetch_ice_servers()

        self.sio = socketio.AsyncClient(
            ssl_verify=False,
            reconnection=True,
            reconnection_attempts=0,
            reconnection_delay=2,
            reconnection_delay_max=10
        )

        @self.sio.event
        async def connect():
            logger.info("Connected to Haven Socket.IO gateway!")
            await self.sio.emit("get-channels")

        @self.sio.event
        async def disconnect():
            logger.warning("Disconnected from Haven Socket.IO gateway.")

        @self.sio.on("channels-list")
        async def on_channels_list(channels):
            for ch in channels:
                code = ch.get("code")
                name = ch.get("name")
                if code and name:
                    self.channel_names[code] = name
                    if ch.get("voice_enabled", 1):
                        await self.sio.emit("request-voice-users", {"code": code})
            logger.info(f"Loaded {len(self.channel_names)} channels list.")

        @self.sio.on("new-message")
        async def on_new_message(data):
            channel_code = data.get("channelCode")
            msg = data.get("message", {})
            author_id = msg.get("author")
            author_name = msg.get("username") or "User"
            content = msg.get("content", "")

            if author_name == BOT_USERNAME or author_id == self.user_id:
                return

            if content.startswith("!"):
                logger.info(f"Command received in {channel_code} from {author_name} (ID: {author_id}): {content}")
                await self.handle_chat_command(channel_code, author_id, author_name, content)

        @self.sio.on("voice-users-update")
        async def on_voice_users_update(data):
            channel_code = data.get("channelCode")
            users = data.get("users", [])
            for k in list(self.user_voice_locations.keys()):
                if self.user_voice_locations[k] == channel_code:
                    del self.user_voice_locations[k]

            for u in users:
                uid = u.get("id")
                uname = u.get("username")
                if uid and uid != self.user_id:
                    self.user_voice_locations[uid] = channel_code
                if uname:
                    self.user_voice_locations[uname.lower()] = channel_code

            logger.info(f"Updated voice locations for {channel_code}: {[u.get('username') for u in users]}")
            if self.in_voice and self.current_voice_code == channel_code:
                self.active_listeners = {u["id"] for u in users if u["id"] != self.user_id}
                self._reset_idle_timer()
                if self.broadcaster.is_playing:
                    await self._ensure_all_listeners_connected()

        @self.sio.on("voice-existing-users")
        async def on_existing_users(data):
            channel_code = data.get("channelCode")
            users = data.get("users", [])
            logger.info(f"voice-existing-users in {channel_code}: {len(users)} users")
            self.active_listeners = {u["id"] for u in users if u["id"] != self.user_id}
            for u in users:
                uid = u.get("id")
                if uid and uid != self.user_id:
                    await self._create_peer_and_offer(uid)
            self._reset_idle_timer()

        @self.sio.on("voice-user-joined")
        async def on_user_joined(data):
            channel_code = data.get("channelCode")
            user = data.get("user", {})
            uid = user.get("id")
            uname = user.get("username")
            if uid and uid != self.user_id:
                logger.info(f"User joined voice {channel_code}: {uname} (ID: {uid})")
                self.user_voice_locations[uid] = channel_code
                if uname:
                    self.user_voice_locations[uname.lower()] = channel_code
                if self.in_voice and self.current_voice_code == channel_code:
                    self.active_listeners.add(uid)
                    self._reset_idle_timer()
                    logger.info(f"Proactively offering audio track to newcomer {uname} ({uid})")
                    await self._create_peer_and_offer(uid)

        @self.sio.on("voice-user-left")
        async def on_user_left(data):
            channel_code = data.get("channelCode")
            user = data.get("user", {})
            uid = user.get("id")
            uname = user.get("username")
            if uid in self.user_voice_locations:
                del self.user_voice_locations[uid]
            if uname and uname.lower() in self.user_voice_locations:
                del self.user_voice_locations[uname.lower()]

            if uid and uid in self.peer_connections:
                logger.info(f"User left voice: ID {uid}")
                pc = self.peer_connections.pop(uid, None)
                asyncio.create_task(self._safe_close_pc(pc, uid))

            if uid in self.active_listeners:
                self.active_listeners.remove(uid)

            if self.in_voice and self.current_voice_code == channel_code:
                self._reset_idle_timer()

        @self.sio.on("voice-offer")
        async def on_voice_offer(data):
            from_user = data.get("from", {})
            target_user_id = from_user.get("id")
            offer_data = data.get("offer", {})
            if not target_user_id or target_user_id == self.user_id:
                return

            logger.info(f"Received voice-offer from {from_user.get('username')} ({target_user_id})")
            old_pc = self.peer_connections.pop(target_user_id, None)
            if old_pc:
                asyncio.create_task(self._safe_close_pc(old_pc, target_user_id))

            pc = RTCPeerConnection(configuration=self.rtc_config)
            self.peer_connections[target_user_id] = pc

            # Dedicated broadcaster track for this peer
            track = self.broadcaster.create_track_for_peer(target_user_id)
            pc.addTrack(track)

            @pc.on("connectionstatechange")
            def on_state():
                logger.info(f"Peer {target_user_id} connection state: {pc.connectionState}")
                if pc.connectionState in ["failed", "closed"]:
                    asyncio.create_task(self._safe_close_pc(pc, target_user_id))
                    self.peer_connections.pop(target_user_id, None)

            offer = RTCSessionDescription(sdp=offer_data["sdp"], type=offer_data["type"])
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            await self.sio.emit("voice-answer", {
                "code": self.current_voice_code,
                "targetUserId": target_user_id,
                "answer": {
                    "type": pc.localDescription.type,
                    "sdp": pc.localDescription.sdp
                }
            })

        @self.sio.on("voice-answer")
        async def on_voice_answer(data):
            from_user = data.get("from", {})
            target_user_id = from_user.get("id")
            answer_data = data.get("answer", {})
            if target_user_id in self.peer_connections:
                logger.info(f"Received voice-answer from {target_user_id}")
                pc = self.peer_connections[target_user_id]
                answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
                await pc.setRemoteDescription(answer)

        @self.sio.on("voice-ice-candidate")
        async def on_remote_ice(data):
            from_user = data.get("from", {})
            target_user_id = from_user.get("id")
            cand_data = data.get("candidate")
            if not cand_data or target_user_id not in self.peer_connections:
                return

            pc = self.peer_connections[target_user_id]
            try:
                raw_cand = cand_data.get("candidate", "")
                if not raw_cand:
                    return
                cand_str = raw_cand.split(":", 1)[1] if ":" in raw_cand else raw_cand
                candidate = candidate_from_sdp(cand_str)
                candidate.sdpMid = cand_data.get("sdpMid")
                candidate.sdpMLineIndex = cand_data.get("sdpMLineIndex")
                await pc.addIceCandidate(candidate)
            except Exception as e:
                logger.debug(f"Failed to add ICE candidate from {target_user_id}: {e}")

        logger.info(f"Connecting to Haven Socket.IO at {HAVEN_URL}...")
        await self.sio.connect(
            HAVEN_URL,
            socketio_path="/socket.io",
            transports=["websocket"],
            auth={"token": self.jwt_token},
            wait_timeout=15
        )

        await self.sio.wait()


if __name__ == "__main__":
    bot = HavenMusicBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Fatal bot crash: {e}", exc_info=True)
