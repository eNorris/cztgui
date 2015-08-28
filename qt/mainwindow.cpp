#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "protocoldialog.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow), db(NULL), model(NULL)
{
    //QLocale::setDefault(QLocale(QLocale::English, QLocale::UnitedStates));
    //QLocale::setDefault(QLocale(QLocale::Chinese, QLocale::China));

    ui->setupUi(this);

    //liu added
    connect(ui->actionLogo, SIGNAL(triggered()), this, SLOT(openweb())); //open the UTS web
    createLanguageMenu(); //liu added dynamically generate multi language menu

    protocolDialog = new ProtocolDialog(this);

    //buildPatientDataBase();
    //db = db_connect("/media/Storage/cztgui/qt/patientdata.db");
    db = db_connect("patientdata.db"); //liu added
    dbFetchPatientInfo();
    buildModel();
    updateSheet();

    connect(ui->tableView, SIGNAL(clicked(QModelIndex)), this, SLOT(updateChildren(QModelIndex)));

    connect(ui->tableView, SIGNAL(activated(QModelIndex)), this, SLOT(updateChildren(QModelIndex)));

    connect(this, SIGNAL(emitUpdateChildren(QModelIndex, QVector<PatientData*>&)), ui->patientInfoForm, SLOT(updateChildren(QModelIndex, QVector<PatientData*>&)));

    //connect(ui->tableView, SIGNAL(clicked(QModelIndex)), pat, SLOT(updateChildren(QModelIndex)));

    //ui->columnView->setModel(&model);

    //    cview->setModel(&model);


}

MainWindow::~MainWindow()
{
    delete ui;
}

// liu added create the menu entries dynamically, dependent on the existing translations.
void MainWindow::createLanguageMenu(void)
{
    QActionGroup* langGroup = new QActionGroup(ui->menuLanguage);
    langGroup->setExclusive(true);

    connect(langGroup, SIGNAL (triggered(QAction *)), this, SLOT (slotLanguageChanged(QAction *)));

    // format systems language
    QString defaultLocale = QLocale::system().name(); // e.g. "en_US"  "zh_CN"
    defaultLocale.truncate(defaultLocale.lastIndexOf('_')); // e.g. "en"

    m_langPath = QApplication::applicationDirPath();
    m_langPath.append("/languages");
    QDir dir(m_langPath);
    QStringList fileNames = dir.entryList(QStringList("cztgui_*.qm"));

    for (int i = 0; i < fileNames.size(); ++i)
    {
        // get locale extracted by filename
        QString locale;
        locale = fileNames[i]; // "cztgui_en.qm"
        locale.truncate(locale.lastIndexOf('.')); // "cztgui_en"
        locale.remove(0, locale.indexOf('_') + 1); // "en"

        QString lang = QLocale::languageToString(QLocale(locale).language());
        QIcon ico(QString("%1/%2.png").arg(m_langPath).arg(locale));
        //qDebug() << lang;

        QAction *action = new QAction(ico, lang, this);
        action->setCheckable(true);
        action->setData(locale);

        ui->menuLanguage->addAction(action);
        langGroup->addAction(action);

        // set default translators and language checked
        if (defaultLocale == locale)
        {
            action->setChecked(true);
        }
    }
}

//liu added  Called every time, when a menu entry of the language menu is called
void MainWindow::slotLanguageChanged(QAction* action)
{
    if(0 != action)
    {
        // load the language dependant on the action content
        loadLanguage(action->data().toString());
        setWindowIcon(action->icon());
    }
}

void switchTranslator(QTranslator& translator, const QString& filename)
{
   // remove the old translator
   qApp->removeTranslator(&translator);

   // load the new translator
   //qDebug() << filename;
   if(translator.load(filename,"languages"))
      qApp->installTranslator(&translator);
   else
      qDebug() << "load qm file failed (" << filename << ")";

}


void MainWindow::loadLanguage(const QString& rLanguage)
{
   if(m_currLang != rLanguage) {
   m_currLang = rLanguage;
   QLocale locale = QLocale(m_currLang);
   QLocale::setDefault(locale);
   QString languageName = QLocale::languageToString(locale.language());
   switchTranslator(m_translator, QString("cztgui_%1.qm").arg(rLanguage));
   //switchTranslator(m_translatorQt, QString("qt_%1.qm").arg(rLanguage));
   ui->statusBar->showMessage(tr("Current Language changed to %1").arg(languageName));
   }
}

/*
void MainWindow::loadLang(const QString& string)
{
    qDebug() << "\n->changeMyLang()";
    if(string == QString("fr"))
    {
        if(!translator->load("app_fr", "../helloLocalization"))
        {
            qDebug() << "Failed to load app_fr translation";
        }
        qApp->installTranslator(translator);
    }

    if(string == QString("sp"))
    {
        if(!translator->load("app_sp", "../helloLocalization"))
        {
            qDebug() << "Failed to load app_sp translation";
        }
        qApp->installTranslator(translator);
    }

    if(string == QString("en"))
    {
        qApp->removeTranslator(translator);
        //qApp->removeTranslator(translator2);
    }
}
*/

void MainWindow::changeEvent(QEvent* event)
{
    if(0 != event) {
        switch(event->type()) {
        // this event is send if a translator is loaded
        case QEvent::LanguageChange:
            ui->retranslateUi(this);
            break;

        // this event is send, if the system, language changes
        //case QEvent::LocaleChange:
        //    QString locale = QLocale::system().name();
        //    locale.truncate(locale.lastIndexOf('_'));
        //    loadLanguage(locale);
        //    break;

        default:
            // Do nothing
            break;
        }
    }
    QMainWindow::changeEvent(event);
}
//liu added end



void MainWindow::updateChildren(QModelIndex indx)
{
    emit emitUpdateChildren(indx, patientVector);
}

void MainWindow::on_acquireProtocolButton_clicked()
{
    int index = ui->otherInfoForm->getProtocolType();

    QString tempstr = ui->otherInfoForm->getProtocolName(); //ui->protocolDropBox->itemText(ci);
    protocolDialog->setProtocolText(tempstr);

    switch(index)
    {
    case 0:
        protocolDialog->setProtocolTime(5.0);
        break;
    case 1:
        protocolDialog->setProtocolTime(30.0);
        break;
    case 2:
        protocolDialog->setProtocolTime(55.5);
        break;
    }

    protocolDialog->exec();
}

QSqlDatabase* MainWindow::db_connect(QString dbname)
{
    if(db != NULL)
    {
        delete db;
        db = NULL;
    }

    //qDebug() << "before creation: " << db;
    db = new QSqlDatabase;
    //qDebug() << "after creation: " << db;

    *db = QSqlDatabase::addDatabase("QSQLITE");
    db->setHostName("localhost");
    db->setDatabaseName(dbname);
    db->setUserName("root");
    db->setPassword("rootpassword");

    //bool ok = db->open();
    //qDebug() << "ok = " << ok;
    //qDebug("%s.", qPrintable(db->lastError().text()));

    return db;
}

void MainWindow::dbFetchPatientInfo()
{
    if(db == NULL)
        qDebug() << "NULL Database!";

    QSqlQuery query(*db);
    bool qgood = query.exec("select * from patient");

    if(qgood)
    {

        while(query.next())
        {
            PatientData *p = new PatientData();
            p->firstName = query.value(0).toString();
            p->middleName = query.value(1).toString();
            p->lastName = query.value(2).toString();
            p->patientId = query.value(3).toInt();
            p->gender = query.value(4).toString();
            p->birthdate = QDate::fromString(query.value(5).toString(), "M/d/yyyy");
            p->weight = query.value(6).toFloat();
            p->height = query.value(7).toFloat();

            patientVector.append(p);
        }
    }
    else
    {
        qDebug() << "Query returned failure!";
        qDebug("%s.", qPrintable(db->lastError().text()));

    }
}

void MainWindow::buildModel()
{
    model = new QStandardItemModel(patientVector.size(),6,this); //2 Rows and 3 Columns
    model->setHorizontalHeaderItem(0, new QStandardItem(QString(tr("Patient Name"))));
    model->setHorizontalHeaderItem(1, new QStandardItem(QString(tr("Patient Id"))));
    model->setHorizontalHeaderItem(2, new QStandardItem(QString(tr("Study Name"))));
    model->setHorizontalHeaderItem(3, new QStandardItem(QString(tr("Start Date/Time"))));
    model->setHorizontalHeaderItem(4, new QStandardItem(QString(tr("Status"))));
    model->setHorizontalHeaderItem(5, new QStandardItem(QString(tr("Accession Number"))));

    ui->tableView->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

}

void MainWindow::updateSheet()
{

    qSort(patientVector.begin(), patientVector.end(), PtrLess<PatientData>());

    for(int i = 0; i < patientVector.size(); i++)
    {
        model->setItem(i, 0, new QStandardItem(patientVector[i]->firstName + " " + patientVector[i]->lastName));
        model->setItem(i, 1, new QStandardItem(QString::number(patientVector[i]->patientId)));
        model->setItem(i, 2, new QStandardItem("Experiment 123"));
        model->setItem(i, 3, new QStandardItem(patientVector[i]->birthdate.toString()));
        model->setItem(i, 4, new QStandardItem("Ongoing"));
        model->setItem(i, 5, new QStandardItem("???"));
    }

    ui->tableView->setModel(model);

}

// liu added
void MainWindow::openweb()
{
    QUrl utsurl("http://ultratech-science.com");
   // qDebug()<<utsurl;
    QDesktopServices::openUrl(utsurl);
}

