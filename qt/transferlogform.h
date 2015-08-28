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

protected:
    //liu added
    // this event is called, when a new translator is loaded or the system language is changed
    void changeEvent(QEvent*); //liu added

};

#endif // TRANSFERLOGFORM_H
