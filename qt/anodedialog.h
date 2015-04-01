#ifndef ANODEDIALOG_H
#define ANODEDIALOG_H

#include <QDialog>

namespace Ui {
class AnodeDialog;
}

class AnodeDialog : public QDialog
{
    Q_OBJECT

public:
    explicit AnodeDialog(QWidget *parent = 0);
    ~AnodeDialog();

private:
    Ui::AnodeDialog *ui;
};

#endif // ANODEDIALOG_H
