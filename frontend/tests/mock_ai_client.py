#!/usr/bin/env python3
"""mock_ai_client.py — Test single-threaded AI client with pselect."""

import sys
import os
import json
import select
import socket
import termios
import errno
import tty
import threading
import http.server

# Add pyterm to path
sys.path.insert(0, "/home/agents/workspace/pyterm/repo")

from pyterm import Terminal

STDIN_FD = 0


class MockAIProvider:
    """Simulates Hermes backend - builds requests without doing I/O."""
    
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
    
    def build_request(self, messages, model="test-model"):
        """Build HTTP request parameters (no I/O)."""
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = json.dumps({
            "model": model,
            "messages": messages,
        })
        return url, headers, body


class MockAIHandler(http.server.BaseHTTPRequestHandler):
    """Simple echo handler that responds with a canned message."""
    
    def do_POST(self):
        # Read request body
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length).decode()
        
        # Respond with mock AI response
        response = {
            "id": "test-123",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Echo: I received your request of {len(body)} bytes"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        
        response_data = json.dumps(response)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(response_data.encode())


def start_mock_server(host='127.0.0.1', port=8888):
    """Start a mock AI server on a separate thread."""
    server = http.server.HTTPServer((host, port), MockAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[mock-server] Listening on {host}:{port}", file=sys.stderr)
    return server


class SingleThreadedAI:
    """Single-threaded AI client using pselect."""
    
    def __init__(self, term, provider):
        self.term = term
        self.provider = provider
        
        # UI state
        self.height = term.height
        self.width = term.width
        self.input_row = self.height - 1
        self.status_row = self.height
        
        # Network state
        self.socket = None
        self.connected = False
        self.recv_buffer = b""
        
        # Input state
        self.input_text = ""
        self.cursor_pos = 0
        
        # History
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            }
        ]
    
    def setup_layout(self):
        """Set up the pyterm UI layout."""
        self.term.clear_screen()
        
        # Create output window
        self.output_win = self.term.window(1, self.height - 2)
        self.output_win.clear()
        
        # Draw input area
        self._draw_input_area()
    
    def _draw_input_area(self):
        """Draw the input box and status bar."""
        prompt_text = f"❯ {self.input_text}"
        self.term.put(prompt_text, 1, self.height - 1)
        
        status = "Connected" if self.connected else "Disconnected"
        self.term.put(f"Status: {status}", 1, self.height)
    
    def connect(self):
        """Connect to AI server."""
        try:
            host, port = "127.0.0.1", 8888
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((host, port))
            self.socket.setblocking(False)
            self.connected = True
            print(f"[net] Connected to {host}:{port}", file=sys.stderr)
        except Exception as e:
            print(f"[error] Connection failed: {e}", file=sys.stderr)
    
    def disconnect(self):
        """Disconnect from AI server."""
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
            self.connected = False
    
    def send_message(self, message_text):
        """Build and send a message to the AI."""
        self.messages.append({
            "role": "user",
            "content": message_text
        })
        
        # Clear input
        self.input_text = ""
        self.cursor_pos = 0
        
        # Show user's message in output
        self.output_win.println(f"[You] {message_text}")
        self._draw_input_area()
        
        if self.connected and self.socket:
            url, headers, body = self.provider.build_request(self.messages)
            
            http_request = f"POST /v1/chat/completions HTTP/1.1\r\n"
            http_request += f"Host: 127.0.0.1:8888\r\n"
            http_request += "Connection: keep-alive\r\n"
            
            for key, value in headers.items():
                http_request += f"{key}: {value}\r\n"
            
            http_request += f"Content-Length: {len(body)}\r\n"
            http_request += "\r\n"
            http_request += body
            
            try:
                self.socket.sendall(http_request.encode())
                print(f"[net] Sent {len(http_request.encode())} bytes", file=sys.stderr)
            except Exception as e:
                print(f"[error] Send failed: {e}", file=sys.stderr)
                self.disconnect()
    
    def handle_response_data(self, data):
        """Process incoming response data."""
        if not data:
            return
        
        self.recv_buffer += data
        
        if b"}" in self.recv_buffer and b"\r\n\r\n" in self.recv_buffer:
            try:
                body_start = self.recv_buffer.index(b"\r\n\r\n") + 4
                body_data = self.recv_buffer[body_start:]
                
                if body_data and body_data[-1] == b'}':
                    response_data = json.loads(body_data.decode())
                    
                    content = response_data['choices'][0]['message']['content']
                    
                    self.messages.append({
                        "role": "assistant",
                        "content": content
                    })
                    
                    self.output_win.println(f"[AI] {content}")
                    self._draw_input_area()
                    
                    self.recv_buffer = b""
            except Exception as e:
                print(f"[warn] Parse error: {e}", file=sys.stderr)
    
    def handle_key(self, key):
        """Handle keyboard input."""
        if key == '\x03':  # Ctrl-C
            return False
        
        elif key in ('\r', '\n'):  # Enter - send message
            if self.input_text.strip():
                self.send_message(self.input_text)
        
        elif key in ('\x7f', '\x08'):  # Backspace
            if self.cursor_pos > 0:
                self.input_text = (self.input_text[:self.cursor_pos-1] + 
                                  self.input_text[self.cursor_pos:])
                self.cursor_pos -= 1
        
        elif len(key) == 1 and key >= ' ':  # Printable character
            self.input_text += key
            self.cursor_pos += 1
        
        self._draw_input_area()
        return True
    
    def run(self):
        """Run the main event loop."""
        self.setup_layout()
        print("\n[ui] Single-threaded AI client started. Ctrl-C to exit.", file=sys.stderr)
        
        self.connect()
        
        while True:
            fds_to_check = [STDIN_FD]
            if self.socket and self.connected:
                fds_to_check.append(self.socket.fileno())
            
            try:
                ready_read, _, _ = select.select(fds_to_check, [], [], 0.5)
            except (select.error, OSError) as e:
                # Handle EINTR - interrupted by signal, just continue
                if hasattr(e, 'errno') and e.errno == errno.EINTR:
                    continue
                raise
            
            # Handle stdin
            if STDIN_FD in ready_read:
                try:
                    key = sys.stdin.read(1)
                    if not self.handle_key(key):
                        break
                except (OSError, IOError):
                    pass
            
            # Handle socket data
            if self.socket and self.connected and self.socket.fileno() in ready_read:
                try:
                    data = self.socket.recv(4096)
                    if data:
                        self.handle_response_data(data)
                    else:
                        print(f"[net] Connection closed by server", file=sys.stderr)
                        self.disconnect()
                except socket.timeout:
                    pass
                except Exception as e:
                    print(f"[error] Socket error: {e}", file=sys.stderr)
                    self.disconnect()


def main():
    # Save original terminal attributes BEFORE any changes
    old_terminal_attrs = termios.tcgetattr(STDIN_FD)
    
    # Create Terminal (this will also save its own state)
    term = Terminal()
    server = None
    ai = None
    
    try:
        # Start mock server
        server = start_mock_server('127.0.0.1', 8888)
        
        # Create provider (simulated backend)
        provider = MockAIProvider(
            base_url="http://127.0.0.1:8888",
            api_key="test-key-12345"
        )
        
        # Create single-threaded AI client
        ai = SingleThreadedAI(term, provider)
        
        # Set up raw input mode
        tty.setraw(STDIN_FD)
        
        # Run event loop
        ai.run()
        
    finally:
        # Restore terminal to EXACTLY the original state we saved
        termios.tcsetattr(STDIN_FD, termios.TCSADRAIN, old_terminal_attrs)
        
        # Clean up resources
        if ai:
            try:
                ai.disconnect()
            except Exception:
                pass
        
        if server:
            try:
                server.shutdown()
            except Exception:
                pass
        
        # Pyterm cleanup (shows cursor, resets colors, etc.)
        try:
            term.cleanup()
        except Exception:
            pass
        
        print("\n[done] Clean exit", file=sys.stderr)


if __name__ == "__main__":
    main()
