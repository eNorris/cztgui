#ifndef SYSTEMFORM_H
#define SYSTEMFORM_H

#include "util.h"

#include <QWidget>
#include <QFileDialog>
#include <QDebug>

#include <QTimer>
#include <QThread>

#include "SimEngine.h"


class ConnectDialog;
class FpgaDialog;
class GlobalConfigDialog;
class AnodeDialog;
class CathodeDialog;
class SystemConfigDialog;

namespace Ui {
    class SystemForm;
}

class SystemForm : public QWidget
{
    Q_OBJECT

public:
    static const int NX = 22;
    static const int NY = 22;

public:
    explicit SystemForm(QWidget *parent = 0);
    ~SystemForm();

    ConnectDialog *connectDialog;
    FpgaDialog *fpgaDialog;
    GlobalConfigDialog *configDialog;
    AnodeDialog *anodeDialog;
    CathodeDialog *cathodeDialog;
    SystemConfigDialog *systemConfigDialog;

private:
    Ui::SystemForm *ui;
    QTimer dataTimer;

    SimEngine *engine;

    const static float FPS_LIMIT = 30.0;
    bool fpsBound;

signals:
    void startRunning(int, float, float);

protected slots:
    void on_browseButton_clicked();
    void on_connectButton_clicked();
    void on_fpgaButton_clicked();
    void on_globalConfigButton_clicked();
    void on_anodeButton_clicked();
    void on_cathodeButton_clicked();
    void on_systemConfigButton_clicked();

    void on_startButton_clicked();
    void on_stopButton_clicked();

    void doAnodeUpdate();
    void doCathodeUpdate();
    void doAnodeCathodeHG(bool hgSet);

public slots:
    void updatesurf(double t, double **data);
    void fpsUnbound();
    void loadDefaults();
};

#endif // SYSTEMFORM_H
