#include "studyinfoform.h"
#include "ui_studyinfoform.h"

StudyInfoForm::StudyInfoForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::StudyInfoForm)
{
    ui->setupUi(this);
}

StudyInfoForm::~StudyInfoForm()
{
    delete ui;
}

// liu added
void StudyInfoForm::changeEvent(QEvent* event)
{
    if(0 != event)
    {
        switch(event->type())
        {
        // this event is send if a translator is loaded
        case QEvent::LanguageChange:
            ui->retranslateUi(this);
            break;

        // this event is send, if the system, language changes
        //  case QEvent::LocaleChange:
        //  {
        //    QString locale = QLocale::system().name();
        //    locale.truncate(locale.lastIndexOf('_'));
        //    loadLanguage(locale);
        //  }
        //  break;

        default:
            // Do nothing
            break;
        }
    }
    QWidget::changeEvent(event);
}
//liu added end
