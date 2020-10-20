## Naming variables in RCX files

RCX files contain tags which can be accessed via python to read from and write to the devices buffer, set the duration of a trial or change the channel though which a signal is sent. Those tags are adressed by their name as a string, so
```
freefield.setup.set_variable("chan", 1)
```
Will set the value of the variable attached to the tag "chan" to 1. For our built in functions to work, it is necessary that the tags follow our naming conventions. For example
```
freefield.setup.wait_to_finish_playing()
```
Assumes that there is a tag with the name "playback" that referes to a variable indicating if something is being played back (playback=1) or not (playback=0). If the tag with that name does not exist the function won't work.
Here is a list of tags that are currently used in toolbox functions:

```
"chan" # int, number of the analog I/O channel.
"playback" # bool, indicates if something is being played back or recorded.
"playbuflen" # int, the duration in samples for which the circuit is running after triggering.
"data" # array of floats, data stored in the memory of the device. Can be read and written to.
```
