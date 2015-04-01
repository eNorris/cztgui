#ifndef TRANSFERLOGFORM_H
#define TRANSFERLOGFORM_H

#include <QWidget>
#include <QStandardItemModel>
#include <QDate>

namespace Ui {
class TransferLogForm;
}

class TransferLogForm : public QWidget
{
    Q_OBJECT

public:
    explicit TransferLogForm(QWidget *parent = 0);
    ~TransferLogForm();

private:
    Ui::TransferLogForm *ui;
};

#endif // TRANSFERLOGFORM_H
