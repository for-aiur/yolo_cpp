#include "yolo.h"
#include <boost/python.hpp>

BOOST_PYTHON_MODULE(yolopy)
{
	using namespace boost::python;
	class_<YoloPython>("YoloPython")
	.def("set_thresh", &YoloPython::setThreshold)
	.def("comp", &YoloPython::getComponent)
	.def("detect", &YoloPython::detect)
	.def("init", &YoloPython::init);
}
