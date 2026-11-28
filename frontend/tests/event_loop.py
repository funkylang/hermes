#!/usr/bin/env python3
"""event_loop.py — Basic pselect-based event loop for pyterm frontend.

This demonstrates the core pattern: using pselect to multiplex between
- stdin (user input)
- AI server responses
- tool output (via helper threads when needed)

Usage pattern:
  while not done:
    ready = pselect([stdin, ai_fd, ...], timeout=0.5s)
    
    if stdin in ready:
      handle_key_input()
      
    if ai_fd in ready:
      handle_ai_response()
"""

import sys
import os
import select
import termios
import tty

# File descriptors for our event sources
STDIN_FD = 0  # Standard input (keyboard)

class EventEmitter:
    """Simple event emitter for communicating between threads and main loop."""
    
    def __init__(self):
        # Create a pipe - we'll write to it from other threads to wake up pselect
        self.read_fd, self.write_fd = os.pipe()
        # Make write end non-blocking
        os.set_blocking(self.write_fd, False)
        
    def notify(self):
        """Signal that an event has occurred (thread-safe)."""
        try:
            os.write(self.write_fd, b'1')
        except (OSError, BlockingIOError):
            pass  # Buffer full or closed - that's okay
            
    def drain(self):
        """Clear any pending notifications."""
        while True:
            try:
                data = os.read(self.read_fd, 1024)
                if not data:
                    break
            except (OSError, BlockingIOError):
                break
                
    def close(self):
        """Close both ends of the pipe."""
        os.close(self.read_fd)
        os.close(self.write_fd)


def setup_raw_input():
    """Set up raw mode stdin for non-blocking reads."""
    tty.setraw(sys.stdin.fileno())
    # Put into non-blocking mode
    fl = termios.tcgetattr(sys.stdin.fileno())
    fl[3] |= os.O_NONBLOCK
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, fl)


def wait_for_events(fds_to_check, timeout=0.5):
    """Wait for events using pselect.
    
    Returns the list of file descriptors that are ready.
    """
    ready_read, _, _ = select.pselect(fds_to_check, None, None, timeout)
    return ready_read


def main():
    # Set up raw input
    setup_raw_input()
    
    # Create event emitters (simulating AI server and tool notifications)
    ai_emitter = EventEmitter()
    tool_emitter = EventEmitter()
    
    print("Event loop started. Press Ctrl-C to quit.\n", file=sys.stderr)
    
    try:
        while True:
            # Wait on: stdin, AI pipe, tool pipe
            fds = [STDIN_FD, ai_emitter.read_fd, tool_emitter.read_fd]
            
            ready = wait_for_events(fds, timeout=0.5)
            
            if not ready:
                continue  # Timeout - can do periodic tasks here
            
            # Handle stdin (user input)
            if STDIN_FD in ready:
                try:
                    key = sys.stdin.read(1)
                    if key == '\x03':  # Ctrl-C
                        print("\nGot Ctrl-C", file=sys.stderr)
                        break
                    elif key:
                        print(f"Got key: {repr(key)}", file=sys.stderr)
                except (OSError, IOError):
                    pass  # Handle non-blocking read errors
            
            # Handle AI server events
            if ai_emitter.read_fd in ready:
                ai_emitter.drain()
                print("AI event detected", file=sys.stderr)
            
            # Handle tool events  
            if tool_emitter.read_fd in ready:
                tool_emitter.drain()
                print("Tool event detected", file=sys.stderr)
    
    finally:
        # Cleanup
        ai_emitter.close()
        tool_emitter.close()
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, 
                         termios.tcgetattr(sys.stdin.fileno()))


if __name__ == "__main__":
    main()
