#include "imagetransferform.h"
#include "ui_imagetransferform.h"

ImageTransferForm::ImageTransferForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ImageTransferForm)
{
    ui->setupUi(this);

    QStandardItemModel *model = new QStandardItemModel(2,5,this); //2 Rows and 5 Columns
    model->setHorizontalHeaderItem(0, new QStandardItem(QString(tr("Patient Name"))));
    model->setHorizontalHeaderItem(1, new QStandardItem(QString(tr("Patient Id"))));
    model->setHorizontalHeaderItem(2, new QStandardItem(QString(tr("Study Name"))));
    model->setHorizontalHeaderItem(3, new QStandardItem(QString(tr("Start Date"))));
    model->setHorizontalHeaderItem(4, new QStandardItem(QString(tr("Study Description"))));
    //model->setHorizontalHeaderItem(5, new QStandardItem(QString("Accession Number")));
    ui->tableView->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    QString s = QDateTime::currentDateTime().toString();
    model->setItem(0, 0, new QStandardItem("Bob"));
    model->setItem(0, 1, new QStandardItem("314159"));
    model->setItem(0, 2, new QStandardItem("Experiment 429"));
    model->setItem(0, 3, new QStandardItem(QDate::currentDate().toString()));
    model->setItem(0, 4, new QStandardItem("Dr. Frankenstein is requesting more power, we may need to use lighting."));
    //model->setItem(0, 5, new QStandardItem("???"));
    //ui->tableView->setModel(model);
    ui->tableView->setModel(model);
}

ImageTransferForm::~ImageTransferForm()
{
    delete ui;
}


// liu added
void ImageTransferForm::changeEvent(QEvent* event)
{
  if(0 != event) {
    switch(event->type()) {
    // this event is send if a translator is loaded
        case QEvent::LanguageChange:
        ui->retranslateUi(this);
        break;
    default:
        // Do nothing
        break;

    // this event is send, if the system, language changes
    //  case QEvent::LocaleChange:
    //  {
    //    QString locale = QLocale::system().name();
    //    locale.truncate(locale.lastIndexOf('_'));
    //    loadLanguage(locale);
    //  }
    //  break;
    }
  }
 QWidget::changeEvent(event);
}
//liu added end
