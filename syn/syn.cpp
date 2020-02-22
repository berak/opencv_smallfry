#include "AudioFile.h"
#include "RtAudio.h"

#include <iostream>
#include <cstdlib>
#include <cmath>

const std::string path = "C:/Windows/Media/";

RtAudio::StreamOptions options;
size_t bufferFrames = 512;
size_t fs = 24000;
float master_vol = 3.0f;

struct msg {
	int c;
	int e;
	float v;
	std::string s;
};

enum Events {
	M_TIM, // time, 0=Note_on, -1=Note_off
	M_VOL, // vol
	M_FRQ, // dx
	M_LOOP,
	M_SND, // change audio file
	M_END,
	M_MASTER
};

struct samp {
	float t, dx, v;
	bool loop;
	AudioFile<float> au;
	std::string fn;
	samp(std::string file, float dx, float v)
		: t(0), dx(dx), v(v), loop(false), fn(file) {
		au.load(path+file);
	}
	float tick() {
		if (t >= 0) {
			t += dx;
			if (t >= au.getNumSamplesPerChannel()) {
				t = loop ? 0 : -1;
			}
		}
		return t;
	}
	float l() {
		if (t < 0) return 0; // OFF
		return v * au.samples[0][(int)t];
	}
	float r() {
		if (t < 0) return 0; // OFF
		return v * au.samples[1][(int)t];
	}
	void parse(const msg &m) {
		if (!loop && (t < 0 || t >= au.getNumSamplesPerChannel())) {
			t = 0;
		}
		switch(m.e) {
			case M_TIM:	/*std::cout << "TIM " << m.v << std::endl;*/ t = m.v; break;
			case M_VOL:	/*std::cout << "VOL " << m.v << std::endl;*/ v = m.v; break;
			case M_FRQ:	/*std::cout << "FRQ " << m.v << std::endl;*/ dx = m.v; break;
			case M_LOOP:/*std::cout << "FRQ " << m.v << std::endl;*/ loop = m.v; break;
			case M_SND:	/*std::cout << "SND " << m.s << std::endl;*/ fn = m.s; au.load(path + m.s); break;
		}
	}
	void dump(int id) {
		fprintf(stderr,"%d %4.2f %2.2f %4.2f %d %s\n",id, t, dx, v, loop, fn.c_str());
	}
};
/*
template <class T>
class voice
{
	T impl;
public:
	voice<T>(const T &t) : impl(t) {}
	float tick() { return impl.tick(); }
	float l() { return impl.l(); }
	float r() { return impl.r(); }
	void parse(const msg &m) { impl.parse(m); }
	void dump(int id) { impl.dump(id); }
};

std::vector<voice<samp> > tone;
*/
std::vector<samp> tone;
// Interleaved buffers
int mixer( void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
         double streamTime, RtAudioStreamStatus status, void *data )
{
	float *buffer = (float *) outputBuffer;
	if ( status )
		std::cerr << "Stream underflow detected: " << status << std::endl;

	for (size_t i=0; i<nBufferFrames; i++ ) {
		float l = 0;
		float r = 0;
		for (size_t j=0; j<tone.size(); j++) {
			tone[j].tick();
			l += tone[j].l();
			r += tone[j].r();
		}
		*buffer++ = (l * master_vol);
		*buffer++ = (r * master_vol);
	}
	return 0;
}


int main( int argc, char *argv[] ) {

	for (int i=1; i<argc; i++) {
		tone.push_back(samp(argv[i], 1, 0.25));
	}

	RtAudio dac;
	if ( dac.getDeviceCount() < 1 ) {
		std::cout << "\nNo audio devices found!\n";
		exit( 1 );
	}

	RtAudio::StreamParameters oParams;
	oParams.nChannels = 2;
	oParams.firstChannel = 0;
	oParams.deviceId = dac.getDefaultOutputDevice();

	options.flags = RTAUDIO_HOG_DEVICE;
	options.flags |= RTAUDIO_SCHEDULE_REALTIME;
	options.flags |= RTAUDIO_MINIMIZE_LATENCY;

	try {
		dac.openStream( &oParams, NULL, RTAUDIO_FLOAT32, fs, &bufferFrames, &mixer, NULL/*(void *)data*/, &options, NULL );
		dac.startStream();
	}
	catch ( RtAudioError& e ) {
		e.printMessage();
		goto cleanup;
	}

	for (int i=0; i<tone.size(); i++) {
		tone[i].dump(i);
	}
	while(true) { // c e (s|v)
	  	msg m;
		std::cin >> m.c;
		if (m.c >= tone.size())
			continue;
		std::cin >> m.e;
		if (m.e == M_MASTER) {
			std::cin >> master_vol;
			continue;
		}
		if (m.e >= M_END)
			break;
		if (m.e == M_SND) {
			std::cin >> m.s;
		} else {
			std::cin >> m.v;
		}
		tone[m.c].parse(m);
	}

	try {
		dac.stopStream();
	}
	catch ( RtAudioError& e ) {
		e.printMessage();
	}

	cleanup:
	if ( dac.isStreamOpen() ) dac.closeStream();

	return 0;
}
