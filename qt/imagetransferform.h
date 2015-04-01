#ifndef IMAGETRANSFERFORM_H
#define IMAGETRANSFERFORM_H

#include <QWidget>
#include <QStandardItemModel>
#include <QDate>

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
};

#endif // IMAGETRANSFERFORM_H
