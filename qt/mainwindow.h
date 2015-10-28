#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QStandardItemModel>
#include <QDateTime>
#include <QVector>

#include <QtSql/QSql>
#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlDriver>
#include <QtSql/QSqlQuery>
#include <QtSql/QSqlError>

#include <QTranslator>
#include <QDesktopServices>
#include <QUrl>

#include "patientdata.h"

class ProtocolDialog;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    //void buildPatientDataBase();

public:
    Ui::MainWindow *ui;

protected:
    // this event is called, when a new translator is loaded or the system language is changed
    void changeEvent(QEvent*); //liu added

    ProtocolDialog *protocolDialog;
    QSqlDatabase *db;
    QStandardItemModel *model;

    QVector<PatientData*> patientVector;

    QSqlDatabase* db_connect(QString dbname);
    void dbFetchPatientInfo();
    void buildModel();
    void updateSheet();

signals:
    void emitUpdateChildren(QModelIndex, QVector<PatientData*>&);

public slots:
    void updateChildren(QModelIndex indx);
    void on_acquireProtocolButton_clicked();
    void openweb();


protected slots: //liu added
  // this slot is called by the language menu actions
    void slotLanguageChanged(QAction* action);

private: //liu added
 // loads a language by the given language shortcur (e.g. cn, en)
    void loadLanguage(const QString& rLanguage);

 // creates the language menu dynamically from the content of m_langPath
    void createLanguageMenu(void);

    QTranslator m_translator; // contains the translations for this application
    QTranslator m_translatorQt; // contains the translations for qt
    QString m_currLang; // contains the currently loaded language
    QString m_langPath; // Path of language files. This is always fixed to /languages.

};

template <typename T>
struct PtrLess // public std::binary_function<bool, const T*, const T*>
{
  bool operator()(const T* a, const T* b) const
  {
    // may want to check that the pointers aren't zero...
    return *a < *b;
  }
};

#endif // MAINWINDOW_H
