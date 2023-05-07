// Geometric Tools, LLC
// Copyright (c) 1998-2013
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// File Version: 5.0.1 (2010/10/01)

#ifndef WM5MINVOLUMEBOX3_H
#define WM5MINVOLUMEBOX3_H

#include "Wm5MathematicsLIB.h"
#include "Wm5Box3.h"
#include "Wm5Query.h"

namespace Wm5
{
// Compute a minimum volume oriented box containing the specified points.
//
// This is a function class.  Use it as follows:
//   Box3<Real> minBox = MinBox3(numPoints, points, epsilon, queryType);

template <typename Real>
class WM5_MATHEMATICS_ITEM SymBox3
{
public:
    //SymBox3 (int numPoints, const Vector3<Real>* points, Real epsilon, Query::Type queryType);
    SymBox3(int numSamples, int numPoints, int numFaces, const Vector3<Real>* samples, const Vector3<Real>* points, const Vector3<Real>* faces, Real epsilon, Query::Type queryType);
    SymBox3(int numSamples, int numPoints, int numFaces, int numOrigFaces, const Vector3<Real>* samples, const Vector3<Real>* points, const Vector3<Real>* faces, const Vector3<Real>* origFaces, Real epsilon, Query::Type queryType);

    double CheckSymPlane(const Vector3<Real>* samples, int ns, const Vector3<Real>* points, int np, const Vector3<Real>* faces, int nf, Vector3<Real>& normal, Vector3<Real>& point);
    double* CheckSymPlanes(const Vector3<Real>* samples, int ns, const Vector3<Real>* points, int np, const Vector3<Real>* faces, int nf, Vector3<Real>& normal1, Vector3<Real>& point1, Vector3<Real>& normal2, Vector3<Real>& point2, Vector3<Real>& normal3, Vector3<Real>& point3);
    
    operator Box3<Real> () const;
    
    double* getScores();
    double* getExtents();
    double* getAxes();
    double* getCenters();
    int getNumBoxes();
    int getProblem();
    
	//int* getHullVerts();

    //Vector3<double>* getNormals();
    //Vector3<double>* getPoints();
    //double* getNormals();
    //double* getPoints();
    //int getNumOfPlanes();
    void freeFields();
    
private:
    Box3<Real> mSymBox;
    
    double* mExtents;
    double* mAxes;
    double* mCenters;
    double* mScores;
    int mNumBoxes;
    int mProblem;
    
	//int* mHullVerts;

    //double* mScores;
    //double* mNormals;
    //double* mPoints;
    //Vector3<double>* mNormals;
    //Vector3<double>* mPoints;
    //int mNumPlanes;
};

}

#endif
