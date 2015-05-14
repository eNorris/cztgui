#ifndef PROTOCOLDIALOG_H
#define PROTOCOLDIALOG_H

#include "util.h"

#include <QDialog>
#include <QWidget>
#include <QFileDialog>
#include <QDebug>

#include <QTimer>
#include <QThread>

#include "SimEngine.h"
#include "clickableplot.h"

class SystemForm;

namespace Ui {
class ProtocolDialog;
}

class ProtocolDialog : public QDialog
{
    Q_OBJECT

public:
    static const int NX = 22;
    static const int NY = 22;

public:
    explicit ProtocolDialog(QWidget *parent = 0);
    ~ProtocolDialog();

    SystemForm *sysForm;

private:
    Ui::ProtocolDialog *ui;
    QTimer dataTimer;

    SimEngine *engine;

    const static float FPS_LIMIT = 30.0;
    bool fpsBound;

signals:
    void startRunning(int, float, float);

protected slots:
    void on_startButton_clicked();

public slots:
    void setProtocolText(QString str);
    void setProtocolTime(const double t);

    void updatesurf(double t, double **data);
    void fpsUnbound();
    void loadDefaults();
};

#endif // PROTOCOLDIALOG_H
