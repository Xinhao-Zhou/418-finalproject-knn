#ifndef DATAPOINT_H
#define DATAPOINT_H


class DataPoint
{
    public:
        DataPoint();
        virtual ~DataPoint();

    int id;
    vector<double> attributes;
    int label;
    protected:

    private:
};

#endif // DATAPOINT_H
