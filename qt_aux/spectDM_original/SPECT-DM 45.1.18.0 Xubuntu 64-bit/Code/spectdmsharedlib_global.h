#ifndef SPECTDMSHAREDLIB_GLOBAL_H
#define SPECTDMSHAREDLIB_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(SPECTDMSHAREDLIB_LIBRARY)
#  define SPECTDMSHAREDLIBSHARED_EXPORT Q_DECL_EXPORT
#  define KROMEK_BUILD TRUE
#else
#  define SPECTDMSHAREDLIBSHARED_EXPORT Q_DECL_IMPORT
#endif

#endif // SPECTDMSHAREDLIB_GLOBAL_H
