#ifndef CATHODEDIALOG_H
#define CATHODEDIALOG_H

#include <QDialog>

namespace Ui {
    class CathodeDialog;
}

class CathodeDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CathodeDialog(QWidget *parent = 0);
    ~CathodeDialog();

private:
    Ui::CathodeDialog *ui;
};

#endif // CATHODEDIALOG_H
