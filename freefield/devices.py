from sys import platform
import numpy as np
from freefield import DATADIR
import os.path
import random
import logging
from collections import Counter
if 'win' in platform:
    import win32com.client
    _win = True
else:
    _win = False
    logging.warning('You seem to not be running windows as your OS...\n'
                    'Working with TDT devices is only supported on Windows!')


class Devices(object):
    """
    Class for handling initialization of and basic input/output to TDT-devices.
    Methods include: initializing devices, writing and reading data, sending
    triggers and halting the devices.
    """

    def __init__(self):
        # TODO: initialize devices when creating class instance
        self._procs = dict()
        self._mode = None
        self._zbus = None

    def initialize_devices(self, device_list, zbus=False, connection='GB'):
        """
        Establish connection to one or several TDT-devices.

        Initialize the devices listed in device_list, which can be a list
        or list of lists. The list / each sublist contains the name and model
        of a device as well as the path to an rcx-file with the circuit that is
        run on the device. Elements must be in order name - model - circuit.
        If zbus is True, initialize the ZBus-interface. If the devices are
        already initialized they are reset

        Args:
            device_list (list or list of lists): each sub-list represents one
                device. Contains name, model and circuit in that order
            zbus (bool): if True, initialize the Zbus interface.

        Examples:
            >>> devs = Devices()
            >>> # initialize a device of model 'RP2', named 'RP2' and load
            >>> # the circuit 'example.rcx'. Also intialize ZBus interface:
            >>> devs.initialize_devices(['RP2', 'RP2', 'example.rcx'], True)
            >>> # initialize two devices of model 'RX8' named 'RX81' and 'RX82'
            >>>devs.initialize_devices(['RX81', 'RX8', 'example.rcx'],
            >>>                        ['RX82', 'RX8', 'example.rcx'])
        """
        # TODO: check if names are unique and id rcx files do exist
        logging.info('Initializing TDT devices, this may take a moment ...')
        models = []
        for name, model, circuit in device_list:
            # advance index if a model appears more then once
            models.append(model)
            index = Counter(models)[model] + 1
            self._procs[name] = self._initialize_proc(model, circuit,
                                                      connection, index)
        if zbus:
            self._zbus = self._initialze_zbus(connection)
        if self._mode is None:
            self._mode = "custom"

    def initialize_default(self, mode='play_rec'):
        """
        Initialize devices in a default configuration.

        This function provides a convenient wrapper for initialize_devices.
        depending on the mode, device names and models and rcx files are chosen
        and initialize_devices is called. The modes cover the core functions
        of the toolbox and include:

        'play_rec': play sounds using two RX8s and record them with a RP2
        'play_birec': same as 'play_rec' but record from 2 microphone channels
        'loctest_freefield': sound localization test under freefield conditions
        'loctest_headphones': localization test with headphones
        'cam_calibration': calibrate cameras for headpose estimation

        Args:
            mode (str): default configuration for initializing devices
        """
        if mode.lower() == 'play_rec':
            device_list = [('RP2', 'RP2',  DATADIR/'rcx'/'rec_buf.rcx'),
                           ('RX81', 'RX8', DATADIR/'rcx'/'play_buf.rcx'),
                           ('RX82', 'RX8', DATADIR/'rcx'/'play_buf.rcx')]
        elif mode.lower() == "play_birec":
            device_list = [('RP2', 'RP2',  DATADIR/'rcx'/'bi_rec_buf.rcx'),
                           ('RX81', 'RX8', DATADIR/'rcx'/'play_buf.rcx'),
                           ('RX82', 'RX8', DATADIR/'rcx'/'play_buf.rcx')]
        elif mode.lower() == "loctest_freefield":
            device_list = [('RP2', 'RP2',  DATADIR/'rcx'/'button.rcx'),
                           ('RX81', 'RX8', DATADIR/'rcx'/'play_buf.rcx'),
                           ('RX82', 'RX8', DATADIR/'rcx'/'play_buf.rcx')]
        elif mode.lower() == "loctest_headphones":
            device_list = [('RP2', 'RP2',  DATADIR/'rcx'/'bi_play_buf.rcx'),
                           ('RX81', 'RX8', DATADIR/'rcx'/'bits.rcx'),
                           ('RX82', 'RX8', DATADIR/'rcx'/'bits.rcx')]
        elif mode.lower() == "cam_calibration":
            device_list = [('RP2', 'RP2',  DATADIR/'rcx'/'button.rcx'),
                           ('RX81', 'RX8', DATADIR/'rcx'/'bits.rcx'),
                           ('RX82', 'RX8', DATADIR/'rcx'/'bits.rcx')]
        else:
            raise ValueError(f'mode {mode} is not a valid input!')
        self._mode = mode
        logging.info(f'set mode to {mode}')
        self.initialize_devices(device_list, True, "GB")

    def write(self, tag, value, procs=['RX81', 'RX82']):
        """
        Write data to device(s).

        Set a tag on one or multiple processors to a given value. Processors
        are adressed by their name (the key in the _procs dictionary). The same
        tag can be set to the same value on multiple processors by giving a
        list of names. One can set multiple tags by giving lists for variable,
        value and procs (procs can be a list of lists, see example).

        This function will call SetTagVal or WriteTagV depending on whether
        value is a single integer or float or an array. If the tag could
        not be set (there are different resons why that might be the case) a
        warning is triggered. CAUTION: If the data type of the value arg does
        not match the data type of the tag, write might be successfull but
        the processor might behave strangely.

        Args:
            tag (str): name of the tag in the rcx-circucit where value is
                written to
            value (int, float, list): value that is written to the tag. Must
                match the data type of the tag.
            procs (str, list): name(s) of the device(s) to write to
        Examples:
            >>> # set the value of tag 'data' to array data on RX81 & RX82 and
            >>> # set the value of tag 'x' to 0 on RP2 :
            >>> settag(['data', 'x'], [data, 0], [['RX81', 'RX82'], 'RP2'])
        """
        if isinstance(tag, list):
            if not len(tag) == len(value) == len(procs):
                raise ValueError("tag, value and procs must be same length!")
        else:
            tag, value = [tag], [value]
            if isinstance(procs, str):
                procs = [[procs]]
        # Check if the processors you want to write to are in _procs
        names = [item for sublist in procs for item in sublist]
        if not set(names).issubset(self._procs.keys()):
            raise ValueError('Can not find some of the specified processors!')
        for t, v, proc in zip(tag, value, procs):
            for p in procs:
                if isinstance(value, (list, np.ndarray)):  # TODO: fix this
                    flag = self._procs[p]._oleobj_.InvokeTypes(
                        15, 0x0, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)),
                        t, 0, v)
                    logging.info(f'Set {tag} on {p}.')
                else:
                    flag = self._procs[p].SetTagVal(t, v)
                    logging.info(f'Set {tag} to {value} on {p}.')
            if flag == 0:
                logging.warning(f'Unable to set tag {tag} on {p}')

    def read(self, tag, n_samples=1, proc='RX8'):
        """
        Read data from device.

        Get the value of a tag from a processor. The number of samples to read
        must be specified, default is 1 which means reading a single float or
        integer value. Unlike in the write method, reading multiple variables
        in one call of the function is not supported.

        Args:
            tag (str): name of the device to write to
            n_samples (int): number of samples to read from device, default=1
        Returns:
            type (int, float, list): value read from the tag
        """
        if n_samples > 1:
            value = np.asarray(self._procs[proc].ReadTagV(tag, 0, n_samples))
        else:
            value = self._procs[proc].GetTagVal(tag)
        logging.info(f'Got {tag} from {proc}.')
        return value

    def halt(self):
        """
        Halt all currently active devices.
        """
        # TODO: can we see if halting was successfull
        for proc_name in self._procs.keys():
            proc = getattr(self._procs, proc_name)
            if hasattr(proc, 'Halt'):
                logging.info(f'Halting {proc_name}.')
                proc.Halt()

    def trigger(self, kind='zBusA', proc=None):
        """
        Send a trigger to the devices.

        Use software or the zBus-interface (must be initialized) to send
        a trigger to devices. The zBus triggers are send to
        all devices by definition. For the software triggers, once has to
        specify the device(s).

        Args:
            kind (str, int): kind of trigger that is send. For zBus triggers
                this can be 'zBusA' or 'zBusB', for software triggers it can
                be any integer.
        """
        if isinstance(kind, (int, float)):
            if not proc:
                raise ValueError('Proc needs to be specified for SoftTrig!')
            self._procs[proc].SoftTrg(kind)
            logging.info(f'SoftTrig {kind} sent to {proc}.')
        elif 'zbus' in kind.lower():
            if self.zbus is not None:
                raise ValueError('ZBus needs to be initialized first!')
            elif kind.lower() == "zbusa":
                self.zbus.zBusTrigA(0, 0, 20)
                logging.info('zBusA trigger sent.')
            elif kind.lower() == "zbusb":
                self.zbus.zBusTrigB(0, 0, 20)
        else:
            raise ValueError("Unknown trigger type! Must be 'soft', "
                             "'zBusA' or 'zBusB'!")

    @staticmethod
    def _initialize_proc(model, circuit, connection, index):
        if _win:
            try:
                RP = win32com.client.Dispatch('RPco.X')
            except win32com.client.pythoncom.com_error as err:
                raise ValueError(err)
        else:
            RP = _COM()
        logging.info(f'Connecting to {model} processor ...')
        connected = 0
        if model.upper() == 'RP2':
            connected = RP.ConnectRP2(connection, index)
        elif model.upper() == 'RX8':
            connected = RP.ConnectRX8(connection, index)
        elif model.upper() == 'RM1':
            connected = RP.ConnectRX8(connection, index)
        elif model.upper() == 'RX6':
            connected = RP.ConnectRX8(connection, index)
        if not connected:
            logging.warning(f'Unable to connect to {model} processor!')
        else:  # connecting was successfull, load circuit
            if not RP.ClearCOF():
                logging.warning('clearing control object file failed')
            if not RP.LoadCOF(circuit):
                logging.warning(f'could not load {circuit}.')
            else:
                logging.info(f'{circuit} loaded!')
            if not RP.Run():
                logging.warning(f'Failed to run {model} processor')
            else:
                logging.info(f'{model} processor is running...')
            return RP

    @staticmethod
    def _initialze_zbus(connection):
        if _win:
            try:
                ZB = win32com.client.Dispatch('ZBUS.x')
            except win32com.client.pythoncom.com_error as err:
                logging.warning(err)
        else:
            ZB = _COM()
        if ZB.ConnectZBUS(connection):
            logging.info('Connected to ZBUS.')
        else:
            logging.warning('Failed to connect to ZBUS.')
        return ZB


class _COM():
    """
    Simulate a TDT processor for testing
    """
    def ConnectRP2(self, connection, index):
        if connection not in ["GB", "USB"]:
            return 0
        if not isinstance(index, int):
            return 0
        else:
            return 1

    def ConnectRX8(self, connection, index):
        if connection not in ["GB", "USB"]:
            return 0
        if not isinstance(index, int):
            return 0
        else:
            return 1

    def ConnectRM1(self, connection, index):
        if connection not in ["GB", "USB"]:
            return 0
        if not isinstance(index, int):
            return 0
        else:
            return 1

    def ConnectRX6(self, connection, index):
        if connection not in ["GB", "USB"]:
            return 0
        if not isinstance(index, int):
            return 0
        else:
            return 1

    def ClearCOF(self):
        return 1

    def LoadCOF(self, circuit):
        if not os.path.isfile(circuit):
            return 0
        else:
            return 1

    def Run(self):
        return 1

    def ConnectZBUS(self, connection):
        if connection not in ["GB", "USB"]:
            return 0
        else:
            return 1

    def Halt(self):
        return 1

    def SetTagVal(self, tag, value):
        if not isinstance(tag, str):
            return 0
        if not isinstance(value, (int, float)):
            return 0
        else:
            return 1

    def ReadTagV(self, tag, nstart, n_samples):
        if not isinstance(tag, str):
            return 0
        if not isinstance(nstart, int):
            return 0
        if not isinstance(nstart, int):
            return 0
        if n_samples == 1:
            return 1
        if n_samples > 1:
            return [random.random() for i in range(n_samples)]
