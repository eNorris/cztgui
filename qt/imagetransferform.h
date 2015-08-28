#ifndef IMAGETRANSFERFORM_H
#define IMAGETRANSFERFORM_H

#include <QWidget>
#include <QStandardItemModel>
#include <QDate>

#include <QTranslator>

namespace Ui {
class ImageTransferForm;
}

class ImageTransferForm : public QWidget
{
    Q_OBJECT

public:
    explicit ImageTransferForm(QWidget *parent = 0);
    ~ImageTransferForm();

private:
    Ui::ImageTransferForm *ui;

protected: //liu added
    // this event is called, when a new translator is loaded or the system language is changed
    void changeEvent(QEvent*); //liu added
};

#endif // IMAGETRANSFERFORM_H
