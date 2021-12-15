from pathlib import Path
import numpy as np
from freefield import DIR
import os.path
import random
import logging
from typing import Union
from collections import Counter
try:
    import win32com.client
except ModuleNotFoundError:
    win32com = None
    logging.warning('Could not import pywin32 - working with TDT device is disabled')


class Processors(object):
    """
    Class for handling the interaction with TDT devices. Usually this is not accessed directly but via the functions
    of the `freefield` module.
    """

    def __init__(self):
        self.processors = dict()
        self.mode = None
        self._zbus = None

    def initialize(self, device, zbus=False, connection='GB'):
        logging.info('Initializing TDT device, this may take a moment ...')
        models = []
        if not all([isinstance(p, list) for p in device]):
            device = [device]  # if a single list was provided, wrap it in another list
        # check if the names given to the devices are unique:
        names = [d[0] for d in device]
        if len(names) != len(set(names)):
            raise KeyError("Every device must be given a unique name!")
        # check if the file(s) exist:
        files = [d[2] for d in device]
        for i, file in enumerate(files):
            if not Path(file).exists():
                if (DIR/"data"/"rcx"/file).exists():
                    device[i][2] = str(DIR/"data"/"rcx"/file)
                else:
                    raise FileNotFoundError(f"{file} does not exist!")
        for name, model, circuit in device:
            # advance index if a model appears more then once
            models.append(model)
            index = Counter(models)[model]
            print(f"initializing {name} of type {model} with index {index}")
            self.processors[name] = self._initialize_proc(model, circuit,
                                                          connection, index)
        if zbus:
            self._zbus = self._initialize_zbus(connection)
        if self.mode is None:
            self.mode = "custom"

    def initialize_default(self, mode):
        if mode.lower() == 'play_rec':
            proc_list = [['RP2', 'RP2',  DIR/'data'/'rcx'/'rec_buf.rcx'],
                         ['RX81', 'RX8', DIR/'data'/'rcx'/'play_buf.rcx'],
                         ['RX82', 'RX8', DIR/'data'/'rcx'/'play_buf.rcx']]
        elif mode.lower() == "play_birec":
            proc_list = [['RP2', 'RP2',  DIR/'data'/'rcx'/'bi_rec_buf.rcx'],
                         ['RX81', 'RX8', DIR/'data'/'rcx'/'play_buf.rcx'],
                         ['RX82', 'RX8', DIR/'data'/'rcx'/'play_buf.rcx']]
        elif mode.lower() == "loctest_freefield":
            proc_list = [['RP2', 'RP2',  DIR/'data'/'rcx'/'button.rcx'],
                         ['RX81', 'RX8', DIR/'data'/'rcx'/'play_buf.rcx'],
                         ['RX82', 'RX8', DIR/'data'/'rcx'/'play_buf.rcx']]
        elif mode.lower() == "loctest_headphones":
            proc_list = [['RP2', 'RP2',  DIR/'data'/'rcx'/'bi_play_buf.rcx'],
                         ['RX81', 'RX8', DIR/'data'/'rcx'/'bits.rcx'],
                         ['RX82', 'RX8', DIR/'data'/'rcx'/'bits.rcx']]
        elif mode.lower() == "cam_calibration":
            proc_list = [['RP2', 'RP2',  DIR/'data'/'rcx'/'button.rcx'],
                           ['RX81', 'RX8', DIR/'data'/'rcx'/'bits.rcx'],
                           ['RX82', 'RX8', DIR/'data'/'rcx'/'bits.rcx']]
        else:
            raise ValueError(f'mode {mode} is not a valid input!')
        self.mode = mode
        logging.info(f'set mode to {mode}')
        self.initialize(proc_list, True, "GB")

    def write(self, tag, value, processors):
        if isinstance(value, (np.int32, np.int64)):
            value = int(value)  # use built-int data type
        if isinstance(processors, str):
            if processors == "RX8s":
                processors = [proc for proc in self.processors.keys() if "RX8" in proc]
            elif processors == "all":
                processors = list(self.processors.keys())
            else:
                processors = [processors]
        # Check if the processors are actually there
        if not set(processors).issubset(self.processors.keys()):
            raise ValueError('Can not find some of the specified device!')
        flag = 0
        for proc in processors:
            if isinstance(value, (list, np.ndarray)):  # TODO: fix this
                value = np.array(value)  # convert to array
                if value.ndim > 1:
                    value = value.flatten()
                flag = self.processors[proc]._oleobj_.InvokeTypes(
                    15, 0x0, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)),
                    tag, 0, value)
                logging.info(f'Set {tag} on {proc}.')
            else:
                flag = self.processors[proc].SetTagVal(tag, value)
                logging.info(f'Set {tag} to {value} on {proc}.')
            if flag == 0:
                logging.warning(f'Unable to set tag {tag} on {proc}')
        return flag

    def read(self, tag, proc, n_samples=1):
        if n_samples > 1:
            value = np.asarray(self.processors[proc].ReadTagV(tag, 0, n_samples))
        else:
            value = self.processors[proc].GetTagVal(tag)
        logging.info(f'Got {tag} from {proc}.')
        return value

    def halt(self):
        # TODO: can we see if halting was successfull
        for proc_name in self.processors.keys():
            proc = self.processors[proc_name]
            if hasattr(proc, 'Halt'):
                logging.info(f'Halting {proc_name}.')
                proc.Halt()

    def trigger(self, kind='zBusA', proc=None):
        if isinstance(kind, int):
            if not proc:
                raise ValueError('Proc needs to be specified for SoftTrig!')
            if not 1<= kind >= 10:
                raise ValueError("software triggers must be between 1 and 10!")
            self.processors[proc].SoftTrg(kind)
            logging.info(f'SoftTrig {kind} sent to {proc}.')
        elif 'zbus' in kind.lower():
            if self._zbus is None:
                raise ValueError('ZBus needs to be initialized first!')
            elif kind.lower() == "zbusa":
                self._zbus.zBusTrigA(0, 0, 20)
                logging.info('zBusA trigger sent.')
            elif kind.lower() == "zbusb":
                self._zbus.zBusTrigB(0, 0, 20)
        else:
            raise ValueError("Unknown trigger type! Must be 'soft', "
                             "'zBusA' or 'zBusB'!")

    @staticmethod
    def _initialize_proc(model: str, circuit: str, connection: str, index: int):
        if win32com is not None:
            try:
                rp = win32com.client.Dispatch('RPco.X')
            except win32com.client.pythoncom.com_error as err:
                raise ValueError(err)
        else:
            rp = _COM()
        logging.info(f'Connecting to {model} processor ...')
        connected = 0
        if model.upper() == 'RP2':
            connected = rp.ConnectRP2(connection, index)
        elif model.upper() == 'RX8':
            connected = rp.ConnectRX8(connection, index)
        elif model.upper() == 'RM1':
            connected = rp.ConnectRX8(connection, index)
        elif model.upper() == 'RX6':
            connected = rp.ConnectRX8(connection, index)
        if not connected:
            logging.warning(f'Unable to connect to {model} processor!')
        else:  # connecting was successful, load circuit
            if not rp.ClearCOF():
                logging.warning('clearing control object file failed')
            if not rp.LoadCOF(circuit):
                logging.warning(f'could not load {circuit}.')
            else:
                logging.info(f'{circuit} loaded!')
            if not rp.Run():
                logging.warning(f'Failed to run {model} processor')
            else:
                logging.info(f'{model} processor is running...')
            return rp

    @staticmethod
    def _initialize_zbus(connection: str = "GB"):
        zb = _COM()
        if win32com is not None:
            try:
                zb = win32com.client.Dispatch('ZBUS.x')
            except win32com.client.pythoncom.com_error as err:
                logging.warning(err)
        if zb.ConnectZBUS(connection):
            logging.info('Connected to ZBUS.')
        else:
            logging.warning('Failed to connect to ZBUS.')
        return zb


class _COM:
    """
    Working with TDT device is only possible on windows machines. This dummy class
    simulates the output of a processor to test code on other operating systems
    """
    @staticmethod
    def ConnectRX8(connection: str, index: int) -> int:
        if connection not in ["GB", "USB"]:
            return 0
        if not isinstance(index, int):
            return 0
        else:
            return 1

    @staticmethod
    def ConnectRP2(connection: str, index: int) -> int:
        if connection not in ["GB", "USB"]:
            return 0
        if not isinstance(index, int):
            return 0
        else:
            return 1

    @staticmethod
    def ConnectRM1(connection: str, index: int) -> int:
        if connection not in ["GB", "USB"]:
            return 0
        if not isinstance(index, int):
            return 0
        else:
            return 1

    @staticmethod
    def ConnectRX6(connection: str, index: int) -> int:
        if connection not in ["GB", "USB"]:
            return 0
        if not isinstance(index, int):
            return 0
        else:
            return 1

    @staticmethod
    def ClearCOF() -> int:
        return 1

    @staticmethod
    def LoadCOF(circuit: str) -> int:
        if not os.path.isfile(circuit):
            return 0
        else:
            return 1

    @staticmethod
    def Run() -> int:
        return 1

    @staticmethod
    def ConnectZBUS(connection: str) -> int:
        if connection not in ["GB", "USB"]:
            return 0
        else:
            return 1

    @staticmethod
    def Halt() -> int:
        return 1

    @staticmethod
    def SetTagVal(tag: str, value: Union[int, float]) -> int:
        if isinstance(value, (np.int32, np.int64)):
            value = int(value)
        if not isinstance(tag, str):
            return 0
        if not isinstance(value, (int, float)):
            return 0
        else:
            return 1

    @staticmethod
    def GetTagVal(tag: str) -> int:
        if tag == "playback":  # return 0 so wait function won't block
            return 0
        if not isinstance(tag, str):
            return 0
        return 1

    @staticmethod
    def ReadTagV(tag: str, n_start: int, n_samples: int) -> Union[int, list]:
        if not isinstance(tag, str):
            return 0
        if not isinstance(n_start, int):
            return 0
        if not isinstance(n_start, int):
            return 0
        if n_samples == 1:
            return 1
        if n_samples > 1:
            return [random.random() for i in range(n_samples)]

    @staticmethod
    def zBusTrigA(rack_num: int, trig_type: int, delay: int) -> int:
        if not isinstance(rack_num, int):
            return 0
        if not isinstance(trig_type, int):
            return 0
        if not isinstance(delay, int):
            return 0
        return 1

    @staticmethod
    def zBusTrigB(rack_num: int, trig_type: int, delay: int) -> int:
        if not isinstance(rack_num, int):
            return 0
        if not isinstance(trig_type, int):
            return 0
        if not isinstance(delay, int):
            return 0
        return 1

    class _oleobj_:
        # this is a hack and should be fixed
        @staticmethod
        def InvokeTypes(arg1, arg2, arg3, arg4, arg5, tag, arg6, value):
            return 1
