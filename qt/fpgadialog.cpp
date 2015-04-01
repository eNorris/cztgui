#include "fpgadialog.h"
#include "ui_fpgadialog.h"

FpgaDialog::FpgaDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::FpgaDialog)
{
    ui->setupUi(this);
}

FpgaDialog::~FpgaDialog()
{
    delete ui;
}
