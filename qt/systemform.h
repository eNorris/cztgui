#ifndef SYSTEMFORM_H
#define SYSTEMFORM_H

#include <QWidget>
#include <QFileDialog>
#include <QDebug>

#include <QTimer>
#include <QThread>

//#include "utils.h"
#include "SimEngine.h"

// From qcustomplot
class QCPColorMap;
class QCPColorScale;
class QCPMarginGroup;

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

    int nx, ny;
    QCPColorMap *colorMap;
    QCPColorScale *colorScale;
    QCPMarginGroup *marginGroup;

    SimEngine *engine;

    const static float FPS_LIMIT = 30.0;
    bool fpsBound;

protected slots:
    void on_browseButton_clicked();
    void on_connectButton_clicked();
    void on_fpgaButton_clicked();
    void on_globalConfigButton_clicked();
    void on_anodeButton_clicked();
    void on_cathodeButton_clicked();
    void on_systemConfigButton_clicked();

public slots:
    void updatesurf(double t, double **data);
    void fpsUnbound();
};

#endif // SYSTEMFORM_H
