"""
CAN bus interface and signal decoding utilities.
Supports both real-time CAN bus reading and offline CSV processing.
"""

import os
import time
import struct
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CANSignalType(Enum):
    """CAN signal data types."""
    UNSIGNED = 0
    SIGNED = 1
    FLOAT = 2
    DOUBLE = 3


@dataclass
class CANSignal:
    """Definition of a CAN signal."""
    name: str
    message_id: int
    start_bit: int
    length: int
    byte_order: str = "little"  # or "big"
    is_signed: bool = False
    factor: float = 1.0
    offset: float = 0.0
    unit: str = ""
    description: str = ""


@dataclass
class CANMessage:
    """CAN message with signals."""
    message_id: int
    timestamp: float
    data: bytes
    signals: Dict[str, float] = field(default_factory=dict)


class CANSignalDecoder:
    """
    Decode raw CAN messages into signal values.
    Supports DBC file parsing and manual signal definitions.
    """
    
    def __init__(self, dbc_file: Optional[str] = None, 
                 signals: Optional[List[CANSignal]] = None):
        """
        Initialize decoder.
        
        Args:
            dbc_file: Path to DBC file (optional)
            signals: List of CANSignal definitions (optional)
        """
        self.signals_by_id = {}  # message_id -> list of signals
        self.signals_by_name = {}  # signal_name -> CANSignal
        
        if dbc_file:
            self.load_dbc(dbc_file)
        elif signals:
            self.add_signals(signals)
    
    def add_signal(self, signal: CANSignal):
        """Add a single signal definition."""
        if signal.message_id not in self.signals_by_id:
            self.signals_by_id[signal.message_id] = []
        self.signals_by_id[signal.message_id].append(signal)
        self.signals_by_name[signal.name] = signal
    
    def add_signals(self, signals: List[CANSignal]):
        """Add multiple signal definitions."""
        for signal in signals:
            self.add_signal(signal)
    
    def load_dbc(self, dbc_file: str):
        """
        Load signal definitions from DBC file.
        Simplified implementation - in practice use cantools or similar.
        """
        # This is a simplified version
        # In production, use: https://github.com/eerimoq/cantools
        logger.warning(f"DBC file loading not fully implemented. Use cantools for full support.")
        
        # For now, we'll define common signals manually
        common_signals = [
            # IMU signals (assuming message ID 0x100)
            CANSignal(name="a_x", message_id=0x100, start_bit=0, length=16, 
                     is_signed=True, factor=0.001, unit="m/s²", 
                     description="Longitudinal acceleration"),
            CANSignal(name="a_y", message_id=0x100, start_bit=16, length=16,
                     is_signed=True, factor=0.001, unit="m/s²",
                     description="Lateral acceleration"),
            CANSignal(name="a_z", message_id=0x100, start_bit=32, length=16,
                     is_signed=True, factor=0.001, unit="m/s²",
                     description="Vertical acceleration"),
            CANSignal(name="omega_x", message_id=0x100, start_bit=48, length=16,
                     is_signed=True, factor=0.001, unit="rad/s",
                     description="Roll rate"),
            CANSignal(name="omega_y", message_id=0x100, start_bit=64, length=16,
                     is_signed=True, factor=0.001, unit="rad/s",
                     description="Pitch rate"),
            CANSignal(name="omega_z", message_id=0x100, start_bit=80, length=16,
                     is_signed=True, factor=0.001, unit="rad/s",
                     description="Yaw rate"),
            
            # Wheel speed signals (message ID 0x200)
            CANSignal(name="omega_FL", message_id=0x200, start_bit=0, length=16,
                     is_signed=False, factor=0.1, unit="rad/s",
                     description="Front-left wheel speed"),
            CANSignal(name="omega_FR", message_id=0x200, start_bit=16, length=16,
                     is_signed=False, factor=0.1, unit="rad/s",
                     description="Front-right wheel speed"),
            CANSignal(name="omega_RL", message_id=0x200, start_bit=32, length=16,
                     is_signed=False, factor=0.1, unit="rad/s",
                     description="Rear-left wheel speed"),
            CANSignal(name="omega_RR", message_id=0x200, start_bit=48, length=16,
                     is_signed=False, factor=0.1, unit="rad/s",
                     description="Rear-right wheel speed"),
            
            # Vehicle state (message ID 0x300)
            CANSignal(name="v_x", message_id=0x300, start_bit=0, length=16,
                     is_signed=False, factor=0.01, unit="m/s",
                     description="Longitudinal velocity"),
            CANSignal(name="steering_angle", message_id=0x300, start_bit=16, length=16,
                     is_signed=True, factor=0.1, offset=0, unit="deg",
                     description="Steering wheel angle"),
            CANSignal(name="brake_pressure", message_id=0x300, start_bit=32, length=16,
                     is_signed=False, factor=0.1, unit="bar",
                     description="Brake pressure"),
            
            # Tire temperatures (message ID 0x400)
            CANSignal(name="tire_temp_FL", message_id=0x400, start_bit=0, length=8,
                     is_signed=False, factor=1.0, offset=-40, unit="°C",
                     description="Front-left tire temperature"),
            CANSignal(name="tire_temp_FR", message_id=0x400, start_bit=8, length=8,
                     is_signed=False, factor=1.0, offset=-40, unit="°C",
                     description="Front-right tire temperature"),
            CANSignal(name="tire_temp_RL", message_id=0x400, start_bit=16, length=8,
                     is_signed=False, factor=1.0, offset=-40, unit="°C",
                     description="Rear-left tire temperature"),
            CANSignal(name="tire_temp_RR", message_id=0x400, start_bit=24, length=8,
                     is_signed=False, factor=1.0, offset=-40, unit="°C",
                     description="Rear-right tire temperature"),
        ]
        
        self.add_signals(common_signals)
    
    def decode_message(self, message: CANMessage) -> CANMessage:
        """
        Decode all signals in a CAN message.
        
        Args:
            message: CANMessage to decode
        
        Returns:
            CANMessage with signals decoded
        """
        if message.message_id not in self.signals_by_id:
            return message
        
        for signal in self.signals_by_id[message.message_id]:
            value = self._decode_signal(message.data, signal)
            message.signals[signal.name] = value
        
        return message
    
    def _decode_signal(self, data: bytes, signal: CANSignal) -> float:
        """
        Decode a single signal from message data.
        
        Args:
            data: Message data bytes
            signal: Signal definition
        
        Returns:
            Decoded signal value
        """
        # Extract bytes containing the signal
        byte_start = signal.start_bit // 8
        byte_end = (signal.start_bit + signal.length - 1) // 8 + 1
        
        if byte_end > len(data):
            logger.warning(f"Signal {signal.name} extends beyond message data")
            return 0.0
        
        # Extract relevant bytes
        signal_bytes = data[byte_start:byte_end]
        
        # Convert to integer
        if signal.byte_order == "little":
            # Little endian
            signal_value = int.from_bytes(signal_bytes, byteorder='little', signed=False)
        else:
            # Big endian
            signal_value = int.from_bytes(signal_bytes, byteorder='big', signed=False)
        
        # Extract the specific bits
        bit_start = signal.start_bit % 8
        bit_mask = (1 << signal.length) - 1
        signal_value = (signal_value >> bit_start) & bit_mask
        
        # Handle signed values
        if signal.is_signed:
            if signal_value >= (1 << (signal.length - 1)):
                signal_value -= (1 << signal.length)
        
        # Apply factor and offset
        signal_value = signal_value * signal.factor + signal.offset
        
        return float(signal_value)
    
    def decode_csv(self, csv_file: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Decode CAN data from CSV file.
        
        Args:
            csv_file: Path to CSV file
            timestamp_col: Name of timestamp column
        
        Returns:
            DataFrame with decoded signals
        """
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # If the CSV already has decoded signals, just return it
        if all(signal.name in df.columns for signal in self.signals_by_name.values()):
            return df
        
        # Otherwise, decode from raw data
        # This would require the CSV to have raw CAN message data
        # For now, we'll assume the CSV already has decoded signals
        logger.warning("CSV decoding assumes pre-decoded signals")
        return df


class CANBusInterface:
    """
    Interface for reading from CAN bus.
    Supports both real-time reading and offline file playback.
    """
    
    def __init__(self, decoder: CANSignalDecoder, 
                 interface: str = "virtual", 
                 channel: str = "can0"):
        """
        Initialize CAN bus interface.
        
        Args:
            decoder: CAN signal decoder
            interface: Type of interface ('virtual', 'socketcan', 'vector')
            channel: CAN channel name
        """
        self.decoder = decoder
        self.interface = interface
        self.channel = channel
        self.is_open = False
    
    def open(self):
        """Open the CAN bus connection."""
        if self.interface == "socketcan":
            try:
                import can
                self.bus = can.interface.Bus(
                    channel=self.channel,
                    interface='socketcan'
                )
                self.is_open = True
                logger.info(f"Opened CAN bus on {self.channel}")
            except ImportError:
                logger.error("python-can not installed. Install with: pip install python-can")
                raise
        elif self.interface == "virtual":
            # Virtual bus for testing
            self.is_open = True
            logger.info("Using virtual CAN bus")
        else:
            logger.error(f"Unsupported interface: {self.interface}")
            raise ValueError(f"Unsupported interface: {self.interface}")
    
    def close(self):
        """Close the CAN bus connection."""
        if self.interface == "socketcan" and hasattr(self, 'bus'):
            self.bus.shutdown()
        self.is_open = False
        logger.info("Closed CAN bus connection")
    
    def read_message(self, timeout: float = 1.0) -> Optional[CANMessage]:
        """
        Read a single CAN message.
        
        Args:
            timeout: Timeout in seconds
        
        Returns:
            Decoded CANMessage or None if timeout
        """
        if not self.is_open:
            logger.error("CAN bus not open")
            return None
        
        if self.interface == "socketcan":
            try:
                msg = self.bus.recv(timeout=timeout)
                if msg is None:
                    return None
                
                # Create CANMessage
                can_msg = CANMessage(
                    message_id=msg.arbitration_id,
                    timestamp=msg.timestamp,
                    data=msg.data
                )
                
                # Decode signals
                can_msg = self.decoder.decode_message(can_msg)
                
                return can_msg
            except Exception as e:
                logger.error(f"Error reading CAN message: {e}")
                return None
        
        elif self.interface == "virtual":
            # For testing: return a dummy message
            time.sleep(0.01)  # Simulate delay
            msg_id = 0x100 + (np.random.randint(0, 4) * 0x100)
            data = bytes(np.random.randint(0, 256, 8))
            can_msg = CANMessage(
                message_id=msg_id,
                timestamp=time.time(),
                data=data
            )
            can_msg = self.decoder.decode_message(can_msg)
            return can_msg
        
        return None
    
    def read_messages(self, num_messages: int = 100, timeout: float = 1.0) -> List[CANMessage]:
        """
        Read multiple CAN messages.
        
        Args:
            num_messages: Number of messages to read
            timeout: Timeout per message in seconds
        
        Returns:
            List of decoded CANMessages
        """
        messages = []
        for _ in range(num_messages):
            msg = self.read_message(timeout)
            if msg is None:
                break
            messages.append(msg)
        return messages
    
    def stream_to_dataframe(self, duration: float = 10.0, 
                           sample_rate: int = 100) -> pd.DataFrame:
        """
        Stream CAN data for a duration and return as DataFrame.
        
        Args:
            duration: Duration to stream in seconds
            sample_rate: Target sample rate in Hz
        
        Returns:
            DataFrame with all signals
        """
        if not self.is_open:
            self.open()
        
        # Calculate number of samples
        num_samples = int(duration * sample_rate)
        interval = 1.0 / sample_rate
        
        # Collect all signals
        all_signals = {}
        timestamps = []
        
        start_time = time.time()
        while time.time() - start_time < duration:
            msg = self.read_message(timeout=interval)
            if msg:
                timestamps.append(msg.timestamp)
                for signal_name, value in msg.signals.items():
                    if signal_name not in all_signals:
                        all_signals[signal_name] = []
                    all_signals[signal_name].append(value)
        
        # Create DataFrame
        df = pd.DataFrame(all_signals)
        df["timestamp"] = timestamps
        
        return df


class CANDataLogger:
    """
    Log CAN data to CSV file for later processing.
    """
    
    def __init__(self, output_file: str, decoder: CANSignalDecoder):
        """
        Initialize data logger.
        
        Args:
            output_file: Path to output CSV file
            decoder: CAN signal decoder
        """
        self.output_file = output_file
        self.decoder = decoder
        self.first_write = True
    
    def log_message(self, message: CANMessage):
        """Log a single CAN message."""
        # Prepare data row
        row = {"timestamp": message.timestamp}
        row.update(message.signals)
        
        # Write to CSV
        mode = 'w' if self.first_write else 'a'
        header = self.first_write
        
        df = pd.DataFrame([row])
        df.to_csv(self.output_file, mode=mode, header=header, index=False)
        
        self.first_write = False
    
    def log_messages(self, messages: List[CANMessage]):
        """Log multiple CAN messages."""
        for msg in messages:
            self.log_message(msg)


# Default signal definitions for common automotive signals
def get_default_signals() -> List[CANSignal]:
    """Get default CAN signal definitions."""
    return [
        # IMU (0x100)
        CANSignal(name="a_x", message_id=0x100, start_bit=0, length=16, is_signed=True, factor=0.001, unit="m/s²"),
        CANSignal(name="a_y", message_id=0x100, start_bit=16, length=16, is_signed=True, factor=0.001, unit="m/s²"),
        CANSignal(name="a_z", message_id=0x100, start_bit=32, length=16, is_signed=True, factor=0.001, unit="m/s²"),
        CANSignal(name="omega_x", message_id=0x100, start_bit=48, length=16, is_signed=True, factor=0.001, unit="rad/s"),
        CANSignal(name="omega_y", message_id=0x100, start_bit=64, length=16, is_signed=True, factor=0.001, unit="rad/s"),
        CANSignal(name="omega_z", message_id=0x100, start_bit=80, length=16, is_signed=True, factor=0.001, unit="rad/s"),
        
        # Wheel speeds (0x200)
        CANSignal(name="omega_FL", message_id=0x200, start_bit=0, length=16, is_signed=False, factor=0.1, unit="rad/s"),
        CANSignal(name="omega_FR", message_id=0x200, start_bit=16, length=16, is_signed=False, factor=0.1, unit="rad/s"),
        CANSignal(name="omega_RL", message_id=0x200, start_bit=32, length=16, is_signed=False, factor=0.1, unit="rad/s"),
        CANSignal(name="omega_RR", message_id=0x200, start_bit=48, length=16, is_signed=False, factor=0.1, unit="rad/s"),
        
        # Vehicle state (0x300)
        CANSignal(name="v_x", message_id=0x300, start_bit=0, length=16, is_signed=False, factor=0.01, unit="m/s"),
        CANSignal(name="steering_angle", message_id=0x300, start_bit=16, length=16, is_signed=True, factor=0.1, unit="deg"),
        CANSignal(name="brake_pressure", message_id=0x300, start_bit=32, length=16, is_signed=False, factor=0.1, unit="bar"),
        CANSignal(name="throttle", message_id=0x300, start_bit=48, length=8, is_signed=False, factor=0.4, unit="%"),
        
        # Tire temperatures (0x400)
        CANSignal(name="tire_temp_FL", message_id=0x400, start_bit=0, length=8, is_signed=False, factor=1.0, offset=-40, unit="°C"),
        CANSignal(name="tire_temp_FR", message_id=0x400, start_bit=8, length=8, is_signed=False, factor=1.0, offset=-40, unit="°C"),
        CANSignal(name="tire_temp_RL", message_id=0x400, start_bit=16, length=8, is_signed=False, factor=1.0, offset=-40, unit="°C"),
        CANSignal(name="tire_temp_RR", message_id=0x400, start_bit=24, length=8, is_signed=False, factor=1.0, offset=-40, unit="°C"),
        
        # Additional sensors (0x500)
        CANSignal(name="yaw_rate", message_id=0x500, start_bit=0, length=16, is_signed=True, factor=0.01, unit="rad/s"),
        CANSignal(name="pitch", message_id=0x500, start_bit=16, length=16, is_signed=True, factor=0.01, unit="rad"),
        CANSignal(name="roll", message_id=0x500, start_bit=32, length=16, is_signed=True, factor=0.01, unit="rad"),
    ]


# Example usage
if __name__ == "__main__":
    # Create decoder with default signals
    signals = get_default_signals()
    decoder = CANSignalDecoder(signals=signals)
    
    # Example: Decode from CSV
    csv_file = "data/external/mendeley_vehicle/can_data.csv"
    if os.path.exists(csv_file):
        df = decoder.decode_csv(csv_file)
        print(f"Loaded {len(df)} CAN messages with {len(df.columns)} signals")
        print(df.head())
    
    # Example: Real-time CAN bus (requires python-can and SocketCAN)
    # interface = CANBusInterface(decoder, interface="socketcan", channel="can0")
    # interface.open()
    # try:
    #     messages = interface.read_messages(num_messages=10)
    #     for msg in messages:
    #         print(f"Message {msg.message_id:03X}: {msg.signals}")
    # finally:
    #     interface.close()

