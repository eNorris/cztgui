
#include "SimEngine.h"


SimEngine::~SimEngine()
{
    if(built)
    {
        for(int i = 0; i < nx; i++)
            delete [] data[i];
        delete data;
    }
}

void SimEngine::build()
{
    if(built)
        return;
    if(nx <= 0 || ny <= 0)
        return;
    data = new double*[nx];
    prevdata = new double*[nx];
    for(int i = 0; i < nx; i++)
    {
        data[i] = new double[ny];
        prevdata[i] = new double[ny];
        for(int j = 0;j < ny; j++)
        {
            data[i][j] = double(rand())/RAND_MAX;
            prevdata[i][j] = data[i][j];
        }
    }

    built = true;
}

void SimEngine::build(const int nx, const int ny)
{
    this->nx = nx;
    this->ny = ny;
    build();
}

void SimEngine::run(int imgct, float exp, float lat)
{
    if(!built)
        return;

    QTime timer;
    timer.start();

    float min = std::numeric_limits<float>::max();
    float max = -min;

    for(int i = 0; i < imgct; i++)
    {
        qDebug() << "Starting photon collection " << i;
        timer.restart();
        SpectDMDll::StartPhotonCollection();
        delay(static_cast<int>(exp * 1000.0));
        qDebug() << "Stopping photon collection " << i;
        SpectDMDll::StopPhotonCollection();
        qDebug() << "Time to collect: " << timer.elapsed();

        qDebug() << "Saving";
        timer.restart();
        QString filename = QString("~/cztgui/data/out%1.csv").arg(i);
        savePhotons(filename.toStdString().c_str());   //"~/cztgui/data/out.csv");
        qDebug() << "loading";
        loadPhotons("/home/stir/cztgui/data/test_100114.csv", data);
        qDebug() << "Time to save & load: " << timer.elapsed();

        QString str = "";
        for(int i = 0; i < ny; i++)
        {
            for(int j = 0; j < nx; j++)
                str.append(QString::number(data[i][j]) + "   ");
            str.append("\n");
        }
        qDebug() << str;

        // Find the min/max values
        for(int i = 0; i < nx; i++)
            for(int j = 0; j < ny; j++)
            {
                if(data[i][j] < min)
                    min = data[i][j];
                if(data[i][j] > max)
                    max = data[i][j];
            }

        qDebug() << "Max = " << max << "  Min = " << min;
        qDebug() << "Uniformity = " << ((max-min)/(max+min));

        qDebug() << "Sending update signal";
        emit update(i, data);
        qDebug() << "Waited";
        delay(static_cast<int>(lat*100.0));
        //SpectDMDll::SaveCollection("");
    }

    return;
}

void SimEngine::stop()
{
    running = false;
}

double& SimEngine::operator()(const int xIndex, const int yIndex)
{
    return data[xIndex][yIndex];
}

/*
void SimEngine::save()
{
    if(SpectDMDll::SaveCollection("~/cztgui/savefile.csv"))
    {
        qDebug() << "Saving packets";
    }
    else
    {
        //QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}
*/

void SimEngine::savePhotons(QString str)
{
    if(!SpectDMDll::SaveCollection(str.toStdString().c_str()))
        qDebug() << "Failed to save: " << SpectDMDll::GetLastError().c_str();
    else
        qDebug() << "SAVED";
}

void SimEngine::loadPhotons(QString str, double **data)
{
    //col 1 - module_number = C{1};
    //col 2 - time_stamp = C{2};
    //col 3 - photon_index = C{3};
    //col 4 - pixel_index = C{4};
    //col 5 - energy channel = C{5};
    //col 6 - penergy = C{6};
    //col 7 - time_detect = C{7};
    //col 8 - ptime_detect = C{8};

    /*
    4,6,0,273,853,1,0,0
    3,26,0,364,976,1,0,0
    3,37,0,366,885,1,0,0
    4,82,0,385,959,1,0,0
    4,90,0,248,777,1,0,0
    4,98,0,362,960,1,0,0
    4,124,0,297,968,1,0,0
    4,131,0,334,968,1,0,0
    3,131,0,386,971,1,0,0
    1,134,0,6,1065,1,0,0
    3,143,0,432,925,1,0,0
    4,157,0,317,945,1,0,0
    3,164,0,415,969,1,0,0
    4,186,0,382,904,1,0,0
    1,191,0,139,1062,1,0,0
    4,222,0,446,1017,1,0,0
    3,257,0,323,965,1,0,0
    4,269,0,428,933,1,0,0
    4,281,0,335,966,1,0,0
    */

    int linect = 0;

    zeroData(); // Remove all current data

    QFile file(str);
    if(!file.open(QIODevice::ReadOnly)) {
        qDebug() << "Couldn't read " << str;
        QMessageBox::information(0, "error", file.errorString());
    }

    QTextStream in(&file);

    qDebug() << "Reading file...";
    while(!in.atEnd()) {
        linect++;
        // Grab each line and split it by commas
        QString line = in.readLine();
        QStringList fields = line.split(",");
        int index = fields[3].toInt();
        data[index % nx][index / ny] += 1.0;
        //model->appendRow(fields);
    }

    qDebug() << "Finished read " << linect << " lines";

    file.close();

    qDebug() << "LOADED";
}

void SimEngine::delay(int millisecs)
{
    QTime dieTime = QTime::currentTime().addMSecs(millisecs);
    while(QTime::currentTime() < dieTime);
        //QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
}

void SimEngine::zeroData()
{
    for(int i = 0; i < nx; i++)
        for(int j = 0; j < ny; j++)
            data[i][j] = 0.0;
}
