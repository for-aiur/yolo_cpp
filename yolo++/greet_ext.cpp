#include "yolo.h"
#include <boost/python.hpp>

BOOST_PYTHON_MODULE(greet_ext)
{
        boost::python::def("greet", greet);
}
