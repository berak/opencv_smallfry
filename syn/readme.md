some weird attempt at real time audio, controlled from camera ;)

* syn.cpp handles RtAudio (sample playback) and some stdin/stdout pipe protocol to control it via pipes (`$ input | syn samp1.wav samp2.wav ... )
* cam.cpp uses opencv to generate x / y control signals (4 patches==4 instruments) (and outputs to std out)
* track.cs is a simple c# pr0g to trigger syn via pipe