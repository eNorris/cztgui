#include "cathodedialog.h"
#include "ui_cathodedialog.h"

CathodeDialog::CathodeDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::CathodeDialog)
{
    ui->setupUi(this);
}

CathodeDialog::~CathodeDialog()
{
    delete ui;
}
