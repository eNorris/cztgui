#include "anodedialog.h"
#include "ui_anodedialog.h"

AnodeDialog::AnodeDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::AnodeDialog)
{
    ui->setupUi(this);
}

AnodeDialog::~AnodeDialog()
{
    delete ui;
}
