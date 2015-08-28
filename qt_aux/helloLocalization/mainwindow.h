#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTranslator>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    //QAction *fileAction;
    //QMenu *menu;
    //QTranslator *translator1;
    //QTranslator *translator2;
    QTranslator *translator;

    //void retranslateUi();

protected:
    void changeEvent(QEvent *event);

public slots:
    void changeMyLang(const QString & string);
};

#endif // MAINWINDOW_H
