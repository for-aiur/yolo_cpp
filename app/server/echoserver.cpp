#include "echoserver.h"
#include "QtWebSockets/qwebsocketserver.h"
#include "QtWebSockets/qwebsocket.h"
#include <QtCore/QDebug>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonDocument>
#include <yolo.h>

QT_USE_NAMESPACE

EchoServer::EchoServer(quint16 port, bool debug, QObject *parent) :
    QObject(parent),
    m_pWebSocketServer(new QWebSocketServer(QStringLiteral("Echo Server"),
                                            QWebSocketServer::NonSecureMode, this)),
    m_debug(debug),
    mode(0)
{
    Yolo yolo;
    if (m_pWebSocketServer->listen(QHostAddress::Any, port)) {
        if (m_debug)
            qDebug() << "Echoserver listening on port" << port;
        connect(m_pWebSocketServer, &QWebSocketServer::newConnection,
                this, &EchoServer::onNewConnection);
        connect(m_pWebSocketServer, &QWebSocketServer::closed, this, &EchoServer::closed);

        timer = new QTimer();
        timer->setInterval(50);
        connect(timer, SIGNAL(timeout()), this, SLOT(broadcastVideo()));
        timer->start();
        capture.open(0);
    }
}

void EchoServer::algo1(cv::Mat& mat)
{
    cv::rectangle(mat,cv::Rect(0,0,50,50), cv::Scalar(255,0,0), 2); 
}

void EchoServer::algo2(cv::Mat& mat)
{
    cv::rectangle(mat,cv::Rect(0,0,75,75), cv::Scalar(0,0,255), 1);
}

void EchoServer::broadcastVideo()
{
    if(!capture.isOpened())
        return;

    capture >> img;

    if(mode==2)
	algo1(img);
    else if(mode==15)
	algo2(img);

    std::vector<uchar> buff;
    cv::imencode(".jpg",img,buff);
    std::string content(buff.begin(), buff.end());
    QByteArray test_img(QByteArray::fromStdString(content));

    QJsonObject json_obj;
    json_obj["meta"] = QJsonArray() << "100" << "false" << "Osman";
    json_obj["data"] = QJsonValue(QLatin1String(test_img.toBase64().data())).toString();
    QJsonDocument doc(json_obj);

    for( int i=0; i<m_clients.count(); ++i )
    {
        m_clients[i]->sendTextMessage(doc.toJson(QJsonDocument::Indented));
    }
}

EchoServer::~EchoServer()
{
    capture.release();
    timer->stop();

    m_pWebSocketServer->close();
    qDeleteAll(m_clients.begin(), m_clients.end());
}

void EchoServer::onNewConnection()
{
    QWebSocket *pSocket = m_pWebSocketServer->nextPendingConnection();

    connect(pSocket, &QWebSocket::textMessageReceived, this, &EchoServer::processTextMessage);
    //connect(pSocket, &QWebSocket::binaryMessageReceived, this, &EchoServer::processBinaryMessage);
    connect(pSocket, &QWebSocket::disconnected, this, &EchoServer::socketDisconnected);

    m_clients << pSocket;
}

void EchoServer::processTextMessage(QString message)
{
    QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());
    if (m_debug)
        qDebug() << "Message received:" << message;
    if (pClient) {
        mode = message.toInt();
        qDebug() << mode;
        if(mode == 2 || mode == 15)
        {
            //activate face recognition
            timer->start();
        }
        else
        {
            timer->stop();
        }

        //pClient->sendTextMessage(message);
        /*
        cv::Mat m = cv::imread("d:/google_code.png");

        std::vector<uchar> buff;
        cv::imencode(".jpg",m,buff);
        std::string content(buff.begin(), buff.end());
        QByteArray test_img(QByteArray::fromStdString(content));

        QJsonObject json_obj;
        json_obj["meta"] = QJsonArray() << "100" << "false" << "Osman";
        json_obj["data"] = QJsonValue(QLatin1String(test_img.toBase64().data())).toString();
        QJsonDocument doc(json_obj);
        pClient->sendTextMessage(doc.toJson(QJsonDocument::Indented));
        */
    }
}

void EchoServer::socketDisconnected()
{
    QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());
    if (m_debug)
        qDebug() << "socketDisconnected:" << pClient;
    if (pClient) {
        m_clients.removeAll(pClient);
        pClient->deleteLater();
    }
}
