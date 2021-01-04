Working with TDT processors
###########################

The loud speakers in our setup (and other things like response boxes or microphones) are operated using
digital signal processors from Tucker-Davis Technologies (TDT).
Wile the low-level assembly code is programmed "at the factory" we can configure the processors by using
TDT's graphical programming interface to create a circuit that is loaded onto the device. The operation defined
by the circuit will be executed when teh device is ran. The figure below shows an example:

.. image:: images/rcx_example.png
  :width: 400
  :alt: Alternative text

This circuit will generate a waveform with a frequency of 1000 Hz and send it to the analog output channel
number one. Since the circuit does not implement any timing, the output will be generated until we terminate the script.
The tag on the left that says "Freq" is a variable that we can access via Python.
If you want to understand more about programming these circuits,
check out the `documentation <https://www.tdt.com/files/manuals/RPvdsEx_Manual.pdf>`_ provided by TDT.


The Processors class
^^^^^^^^^^^^^^^^^^^^
In **freefield** the processors and their circuits and variables are handled with the :class:`Processors` class.
All active processors are handled in one instance of this class. When initializing the


As mentioned before, to initialize a device one has to specify a circuit when initializing a processor (a few frequently
used circuits are stored in freefield/data/rcx/). Additionally, we have to give a name to the device and specify it's
model as well as whether the device is connected via USB or fibre optical cable.

.. ipython::

  In [1]: from freefield import Processors, DIR;

  In [2]: circuit = str(DIR/"data"/"rcx"/"play_buf.rcx")  # example circuit

  In [3]: my_proc = Processors()  # create a new instance

  In [4]: my_proc.initialize([("my_RX8", "RX8", circuit)])

You can now read from and write to the device.


.. _tag-guidelines:

Guidelines for Naming Tags
^^^^^^^^^^^^^^^^^^^^^^^^^^


