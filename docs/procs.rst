Working with TDT processors
###########################

The loud speakers in our setup (and other things like response boxes or microphones) are operated using
digital signal processors from Tucker-Davis Technologies (TDT).
Wile the low-level assembly code is programmed "at the factory" we can configure the processors by using
TDT's graphical programming interface to create a circuit which is saved in the .rcx format and
loaded onto the device. The operation defined
by the circuit will be executed when teh device is ran. The figure below shows an example:

.. image:: images/rcx_example.png
  :width: 400
  :alt: Alternative text

This circuit will generate a waveform with a frequency of 1000 Hz and send it to the analog output channel
number one. Since the circuit does not implement any timing, the output will be generated until we terminate the script.
The tag on the left that says "Freq" is a variable that we can access via Python.
If you want to understand more about programming these circuits,
check out the `documentation <https://www.tdt.com/files/manuals/RPvdsEx_Manual.pdf>`_ provided by TDT. If you
downloaded the TDT software (see :ref:`Installation`) you can check put some examples in the freefield/data/rcx/ folder.

The Processors class
^^^^^^^^^^^^^^^^^^^^
In **freefield** the processors and their circuits and variables are handled with the :class:`Processors` class.
All active processors are handled in one instance of this class. When initializing, you have to define the name, model
and path to the .rcx file of the processor and pass them in a list. You also have to specify whether the device is
connected via USB or optical cable.

.. ipython::

  In [1]: from freefield import Processors, DIR;

  In [2]: circuit = str(DIR/"data"/"rcx"/"play_buf.rcx")  # example circuit

  In [3]: my_proc = Processors()  # create a new instance

  In [4]: my_proc.initialize(proc_list=["my_RX8", "RX8", circuit], connection="USB")

We now have and instance of :class:`Processors` that contains one RX8-processor with the name "my_RX8" that is
connected via USB. You can also initialize multiple processors in one call of the :meth:`initialize` method
by passing a list of lists (calling the method again will overwrite any previously initialized devices):

.. ipython::

  In [5]: proc_list = [['RP2', 'RP2',  DIR/'data'/'rcx'/'button.rcx'],
     ...:['RX81', 'RX8', DIR/'data'/'rcx'/'play_buf.rcx'],
     ...:['RX82', 'RX8', DIR/'data'/'rcx'/'play_buf.rcx']]

  In [6]: my_proc.initialize(proc_list=proc_list, connection="USB")

The above circuits enable us to write signal to the processors, play them and capture the subjects response using
a button box. Since this is the standard setting for our sound localization test, you can use the :meth:`initialize_default`
method. The above example is synonymous to:

.. ipython::

  In [7]: my_proc.initialize_default(mode="loctest_freefield")

All possible modes are listed in the methods documentation.



.. _tag-guidelines:

Guidelines for Naming Tags
^^^^^^^^^^^^^^^^^^^^^^^^^^


