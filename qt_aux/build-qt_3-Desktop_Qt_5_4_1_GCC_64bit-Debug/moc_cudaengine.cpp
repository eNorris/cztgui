/****************************************************************************
** Meta object code from reading C++ file 'cudaengine.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.4.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../plotengine/cudaengine.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'cudaengine.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.4.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_CudaEngine_t {
    QByteArrayData data[14];
    char stringdata[101];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_CudaEngine_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_CudaEngine_t qt_meta_stringdata_CudaEngine = {
    {
QT_MOC_LITERAL(0, 0, 10), // "CudaEngine"
QT_MOC_LITERAL(1, 11, 8), // "finished"
QT_MOC_LITERAL(2, 20, 0), // ""
QT_MOC_LITERAL(3, 21, 6), // "update"
QT_MOC_LITERAL(4, 28, 8), // "double**"
QT_MOC_LITERAL(5, 37, 14), // "unboundSimRate"
QT_MOC_LITERAL(6, 52, 10), // "adddiffuse"
QT_MOC_LITERAL(7, 63, 1), // "x"
QT_MOC_LITERAL(8, 65, 1), // "y"
QT_MOC_LITERAL(9, 67, 10), // "subdiffuse"
QT_MOC_LITERAL(10, 78, 11), // "setPressure"
QT_MOC_LITERAL(11, 90, 1), // "p"
QT_MOC_LITERAL(12, 92, 4), // "stop"
QT_MOC_LITERAL(13, 97, 3) // "run"

    },
    "CudaEngine\0finished\0\0update\0double**\0"
    "unboundSimRate\0adddiffuse\0x\0y\0subdiffuse\0"
    "setPressure\0p\0stop\0run"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_CudaEngine[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   54,    2, 0x06 /* Public */,
       3,    2,   55,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       5,    0,   60,    2, 0x09 /* Protected */,
       6,    2,   61,    2, 0x0a /* Public */,
       9,    2,   66,    2, 0x0a /* Public */,
      10,    1,   71,    2, 0x0a /* Public */,
      12,    0,   74,    2, 0x0a /* Public */,
      13,    0,   75,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double, 0x80000000 | 4,    2,    2,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    7,    8,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    7,    8,
    QMetaType::Void, QMetaType::Double,   11,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void CudaEngine::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        CudaEngine *_t = static_cast<CudaEngine *>(_o);
        switch (_id) {
        case 0: _t->finished(); break;
        case 1: _t->update((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double**(*)>(_a[2]))); break;
        case 2: _t->unboundSimRate(); break;
        case 3: _t->adddiffuse((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 4: _t->subdiffuse((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 5: _t->setPressure((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 6: _t->stop(); break;
        case 7: _t->run(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (CudaEngine::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&CudaEngine::finished)) {
                *result = 0;
            }
        }
        {
            typedef void (CudaEngine::*_t)(double , double * * );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&CudaEngine::update)) {
                *result = 1;
            }
        }
    }
}

const QMetaObject CudaEngine::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_CudaEngine.data,
      qt_meta_data_CudaEngine,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *CudaEngine::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *CudaEngine::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_CudaEngine.stringdata))
        return static_cast<void*>(const_cast< CudaEngine*>(this));
    return QObject::qt_metacast(_clname);
}

int CudaEngine::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 8;
    }
    return _id;
}

// SIGNAL 0
void CudaEngine::finished()
{
    QMetaObject::activate(this, &staticMetaObject, 0, Q_NULLPTR);
}

// SIGNAL 1
void CudaEngine::update(double _t1, double * * _t2)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_END_MOC_NAMESPACE
