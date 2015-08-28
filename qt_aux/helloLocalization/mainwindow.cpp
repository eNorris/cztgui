#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDebug>
#include <QDialog>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //translator1 = new QTranslator(this);
    //translator2 = new QTranslator(this);
    translator = new QTranslator(this);

    connect(ui->languageComboBox, SIGNAL(activated(const QString &)), this, SLOT(changeMyLang(const QString &)));
    ui->retranslateUi(this);
}

/*
void MainWindow::retranslateUi()
{
    ui->retranslateUi(this);
    qDebug() << "->retranslateUi";
    //ui->menuFile->setTitle(tr("File"));
    //ui->actionFirst_Action->setText(tr("First Action"));
    //ui->outputLabel->setText(tr("My translated text"));
}
*/

void MainWindow::changeEvent(QEvent *event)
{
    qDebug() << "->changeEvent()";
    if(event->type() == QEvent::LanguageChange)
    {
        //retranslateUi();
        ui->retranslateUi(this);
    }
    QMainWindow::changeEvent(event);
}

void MainWindow::changeMyLang(const QString & string)
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

MainWindow::~MainWindow()
{
    delete ui;
}
