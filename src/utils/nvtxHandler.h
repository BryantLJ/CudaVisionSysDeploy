/*
 * nvtxHandler.h
 *
 *  Created on: Oct 28, 2015
 *      Author: Víctor Campmany - vcampmany@cvc.uab.es
 */

#ifndef NVTXHANDLER_H_
#define NVTXHANDLER_H_

/* Include library search path (-L) and library (-l) */
#include <nvToolsExtCuda.h>
#include <nvToolsExtCudaRt.h>

// Colors to paint the traces
#define COLOR_RED 		0xFF4500
#define COLOR_BLUE		0x00008B
#define COLOR_GREEN		0x32CD32
#define COLOR_YELLOW	0xFFFF00
#define COLOR_ORANGE	0xFFA500
#define COLOR_GRAY		0x2F4F4F

class NVTXhandler {
private:
	nvtxRangeId_t 			id;				// Identificator of the trace
	nvtxEventAttributes_t 	eventAttrib;	// Attributes of the trace

public:
	NVTXhandler(int color, const char *message)
	{
		this->id = 0;
		this->eventAttrib = {0};
		this->eventAttrib.version = NVTX_VERSION;
		this->eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
		this->eventAttrib.colorType = NVTX_COLOR_ARGB;
		this->eventAttrib.color = color;
		this->eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
		this->eventAttrib.message.ascii = message;
	}

	inline void nvtxStartEvent()
	{
		this->id = nvtxRangeStartEx(&(this->eventAttrib));
	}
	inline void nvtxStopEvent()
	{
		nvtxRangeEnd(this->id);
	}
	inline void finish()
	{
		this->~NVTXhandler();
	}
	~NVTXhandler()
	{

	}

};


#endif /* NVTXHANDLER_H_ */
