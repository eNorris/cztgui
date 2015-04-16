/****************************************************************************
** Meta object code from reading C++ file 'VerticalAutoslider.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.4.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../plotengine/VerticalAutoslider.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'VerticalAutoslider.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.4.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_VerticalAutoSlider_t {
    QByteArrayData data[5];
    char stringdata[59];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_VerticalAutoSlider_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_VerticalAutoSlider_t qt_meta_stringdata_VerticalAutoSlider = {
    {
QT_MOC_LITERAL(0, 0, 18), // "VerticalAutoSlider"
QT_MOC_LITERAL(1, 19, 11), // "newpressure"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 10), // "autoscroll"
QT_MOC_LITERAL(4, 43, 15) // "autoscrollscale"

    },
    "VerticalAutoSlider\0newpressure\0\0"
    "autoscroll\0autoscrollscale"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_VerticalAutoSlider[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   29,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       3,    1,   32,    2, 0x08 /* Private */,
       4,    1,   35,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::Double,    2,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    2,
    QMetaType::Void, QMetaType::Int,    2,

       0        // eod
};

void VerticalAutoSlider::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        VerticalAutoSlider *_t = static_cast<VerticalAutoSlider *>(_o);
        switch (_id) {
        case 0: _t->newpressure((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 1: _t->autoscroll((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->autoscrollscale((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (VerticalAutoSlider::*_t)(double );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&VerticalAutoSlider::newpressure)) {
                *result = 0;
            }
        }
    }
}

const QMetaObject VerticalAutoSlider::staticMetaObject = {
    { &QSlider::staticMetaObject, qt_meta_stringdata_VerticalAutoSlider.data,
      qt_meta_data_VerticalAutoSlider,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *VerticalAutoSlider::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *VerticalAutoSlider::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_VerticalAutoSlider.stringdata))
        return static_cast<void*>(const_cast< VerticalAutoSlider*>(this));
    return QSlider::qt_metacast(_clname);
}

int VerticalAutoSlider::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QSlider::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 3)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 3;
    }
    return _id;
}

// SIGNAL 0
void VerticalAutoSlider::newpressure(double _t1)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
