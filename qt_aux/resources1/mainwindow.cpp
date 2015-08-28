#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QGraphicsPixmapItem>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QGraphicsScene *scene = new QGraphicsScene;
    ui->graphicsView->setScene(scene);
    //QGraphicsView view(&scene);
    QGraphicsPixmapItem *item = new QGraphicsPixmapItem(QPixmap("/media/Storage/cztgui/qt_aux/resources1/images/balloons.png"));
    QGraphicsRectItem *imgItem = new QGraphicsRectItem(0, 0, 60, 60);\

    QImage *imgOrig = new QImage("/media/Storage/cztgui/qt_aux/resources1/images/earth.png");
    QImage *imgSmall = new QImage(imgOrig->scaled(60, 60, Qt::KeepAspectRatio, Qt::SmoothTransformation));

    imgItem->setFlag(QGraphicsItem::ItemIsMovable, true);

    QBrush brush(*imgSmall);
    imgItem->setBrush(brush);

    //connect(imgItem, SIGNAL(), this, SLOT());

    scene->addItem(item);
    scene->addItem(imgItem);
}

MainWindow::~MainWindow()
{
    delete ui;
}
