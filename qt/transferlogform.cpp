#include "transferlogform.h"
#include "ui_transferlogform.h"

TransferLogForm::TransferLogForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::TransferLogForm)
{
    ui->setupUi(this);

    QStandardItemModel *model = new QStandardItemModel(2,6,this); //2 Rows and 6 Columns
    model->setHorizontalHeaderItem(0, new QStandardItem(QString(tr("Study/Series/Image"))));
    model->setHorizontalHeaderItem(1, new QStandardItem(QString(tr("Name"))));
    model->setHorizontalHeaderItem(2, new QStandardItem(QString(tr("Date"))));
    model->setHorizontalHeaderItem(3, new QStandardItem(QString(tr("From"))));
    model->setHorizontalHeaderItem(4, new QStandardItem(QString(tr("To"))));
    model->setHorizontalHeaderItem(5, new QStandardItem(QString(tr("Progress"))));
    ui->tableView->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    //QString s = QDateTime::currentDateTime().toString();
    model->setItem(0, 0, new QStandardItem("???"));
    model->setItem(0, 1, new QStandardItem("Joe"));
    model->setItem(0, 2, new QStandardItem(QDate::currentDate().toString()));
    model->setItem(0, 3, new QStandardItem("Me"));
    model->setItem(0, 4, new QStandardItem("You"));
    model->setItem(0, 5, new QStandardItem("Uploading..."));

    ui->tableView->setModel(model);
}

TransferLogForm::~TransferLogForm()
{
    delete ui;
}

// liu added
void TransferLogForm::changeEvent(QEvent* event)
{
    if(0 != event)
    {
        switch(event->type()) {
        // this event is send if a translator is loaded
            case QEvent::LanguageChange:
            ui->retranslateUi(this);
            break;

        default:
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
