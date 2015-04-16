/****************************************************************************
** Meta object code from reading C++ file 'SimEngine.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.4.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../plotengine/SimEngine.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'SimEngine.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.4.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_SimEngine_t {
    QByteArrayData data[13];
    char stringdata[85];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_SimEngine_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_SimEngine_t qt_meta_stringdata_SimEngine = {
    {
QT_MOC_LITERAL(0, 0, 9), // "SimEngine"
QT_MOC_LITERAL(1, 10, 8), // "finished"
QT_MOC_LITERAL(2, 19, 0), // ""
QT_MOC_LITERAL(3, 20, 6), // "update"
QT_MOC_LITERAL(4, 27, 8), // "double**"
QT_MOC_LITERAL(5, 36, 10), // "adddiffuse"
QT_MOC_LITERAL(6, 47, 1), // "x"
QT_MOC_LITERAL(7, 49, 1), // "y"
QT_MOC_LITERAL(8, 51, 10), // "subdiffuse"
QT_MOC_LITERAL(9, 62, 11), // "setPressure"
QT_MOC_LITERAL(10, 74, 1), // "p"
QT_MOC_LITERAL(11, 76, 4), // "stop"
QT_MOC_LITERAL(12, 81, 3) // "run"

    },
    "SimEngine\0finished\0\0update\0double**\0"
    "adddiffuse\0x\0y\0subdiffuse\0setPressure\0"
    "p\0stop\0run"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_SimEngine[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   49,    2, 0x06 /* Public */,
       3,    2,   50,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       5,    2,   55,    2, 0x0a /* Public */,
       8,    2,   60,    2, 0x0a /* Public */,
       9,    1,   65,    2, 0x0a /* Public */,
      11,    0,   68,    2, 0x0a /* Public */,
      12,    0,   69,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double, 0x80000000 | 4,    2,    2,

 // slots: parameters
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    6,    7,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    6,    7,
    QMetaType::Void, QMetaType::Double,   10,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void SimEngine::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        SimEngine *_t = static_cast<SimEngine *>(_o);
        switch (_id) {
        case 0: _t->finished(); break;
        case 1: _t->update((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double**(*)>(_a[2]))); break;
        case 2: _t->adddiffuse((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 3: _t->subdiffuse((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 4: _t->setPressure((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 5: _t->stop(); break;
        case 6: _t->run(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (SimEngine::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&SimEngine::finished)) {
                *result = 0;
            }
        }
        {
            typedef void (SimEngine::*_t)(double , double * * );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&SimEngine::update)) {
                *result = 1;
            }
        }
    }
}

const QMetaObject SimEngine::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_SimEngine.data,
      qt_meta_data_SimEngine,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *SimEngine::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *SimEngine::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_SimEngine.stringdata))
        return static_cast<void*>(const_cast< SimEngine*>(this));
    return QObject::qt_metacast(_clname);
}

int SimEngine::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 7;
    }
    return _id;
}

// SIGNAL 0
void SimEngine::finished()
{
    QMetaObject::activate(this, &staticMetaObject, 0, Q_NULLPTR);
}

// SIGNAL 1
void SimEngine::update(double _t1, double * * _t2)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_END_MOC_NAMESPACE
