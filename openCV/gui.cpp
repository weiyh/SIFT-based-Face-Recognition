#include <QtGui>
#include "gui.h"
 
Demo2::Demo2(QWidget *parent) : QWidget(parent) {
    // �N�����ܼ��ݩʪ�l��
    b1 = new QLabel(tr("QLabel"));
    b2 = new QPushButton(tr("QPushButton"));
    b3 = new QCheckBox(tr("QCheckBox"));
    b4 = new QRadioButton(tr("QRadioButton"));
    b5 = new QLineEdit;
     
    // �إߪ����˦�����
    QVBoxLayout *layout = new QVBoxLayout;
    layout->addWidget(b1);
    layout->addWidget(b2);
    layout->addWidget(b3);
    layout->addWidget(b4);
    layout->addWidget(b5);
     
    // �]�w�����˦��P�������D
    setLayout(layout);
    setWindowTitle(tr("Demo2"));
}