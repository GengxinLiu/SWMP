// Geometric Tools, LLC
// Copyright (c) 1998-2013
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// File Version: 5.0.1 (2010/10/01)

#include "Wm5MathematicsPCH.h"
#include "Wm5ContSymBox3.h"
#include "Wm5ContMinBox2.h"
#include "Wm5ConvexHull3.h"
#include "Wm5Plane3.h"
#include "Wm5DistPoint3Triangle3.h"
#include "Wm5EdgeKey.h"
#include "Wm5Memory.h"
#include <stdlib.h>
#include <time.h>


static int compare (const void * a, const void * b)
{
  if (*(double*)a > *(double*)b) return 1;
  else if (*(double*)a < *(double*)b) return -1;
  else return 0;  
}

namespace Wm5
{
//----------------------------------------------------------------------------
template <typename Real>
SymBox3<Real>::SymBox3 (int numSamples, int numPoints, int numFaces, int numOrigFaces, const Vector3<Real>* samples, const Vector3<Real>* points, const Vector3<Real>* faces, const Vector3<Real>* origFaces, Real epsilon, Query::Type queryType)
{
    
	srand (time(NULL));

    mNumBoxes = 0;
    mProblem = 0;
    
	double xs = 0;
	double ys = 0;
	double zs = 0;

	for(int i = 0; i < numPoints; i++)
	{
		xs = xs + points[i].X();
		ys = ys + points[i].Y();
		zs = zs + points[i].Z();
	}

	double massCenterX = xs / numPoints;
	double massCenterY = ys / numPoints;
	double massCenterZ = zs / numPoints;

	Vector3<Real> massCenter(massCenterX, massCenterY, massCenterZ);


    // Get the convex hull of the points.
    ConvexHull3<Real> kHull(numPoints,(Vector3<Real>*)points, epsilon, false, queryType);
    int hullDim = kHull.GetDimension();

    if (hullDim == 0)
    {
        mSymBox.Center = points[0];
        mSymBox.Axis[0] = Vector3<Real>::UNIT_X;
        mSymBox.Axis[1] = Vector3<Real>::UNIT_Y;
        mSymBox.Axis[2] = Vector3<Real>::UNIT_Z;
        mSymBox.Extent[0] = (Real)0;
        mSymBox.Extent[1] = (Real)0;
        mSymBox.Extent[2] = (Real)0;
        
        //mProblem = 1;
        return;
    }

    if (hullDim == 1)
    {
        ConvexHull1<Real>* pkHull1 = kHull.GetConvexHull1();
        const int* hullIndices = pkHull1->GetIndices();

        mSymBox.Center =
            ((Real)0.5)*(points[hullIndices[0]] + points[hullIndices[1]]);
        Vector3<Real> diff =
            points[hullIndices[1]] - points[hullIndices[0]];
        mSymBox.Extent[0] = ((Real)0.5)*diff.Normalize();
        mSymBox.Extent[1] = (Real)0;
        mSymBox.Extent[2] = (Real)0;
        mSymBox.Axis[0] = diff;
        Vector3<Real>::GenerateComplementBasis(mSymBox.Axis[1],
            mSymBox.Axis[2], mSymBox.Axis[0]);

        delete0(pkHull1);
        return;
    }

    int i, j;
    Vector3<Real> origin, diff, U, V, W;
    Vector2<Real>* points2;
    Box2<Real> box2;

    if (hullDim == 2)
    {
        // When ConvexHull3 reports that the point set is 2-dimensional, the
        // caller is responsible for projecting the points onto a plane and
        // calling ConvexHull2.  ConvexHull3 does provide information about
        // the plane of the points.  In this application, we need only
        // project the input points onto that plane and call ContMinBox in
        // two dimensions.

        // Get a coordinate system relative to the plane of the points.
        origin = kHull.GetPlaneOrigin();
        W = kHull.GetPlaneDirection(0).Cross(kHull.GetPlaneDirection(1));
        Vector3<Real>::GenerateComplementBasis(U, V, W);

        // Project the input points onto the plane.
        points2 = new1<Vector2<Real> >(numPoints);
        for (i = 0; i < numPoints; ++i)
        {
            diff = points[i] - origin;
            points2[i].X() = U.Dot(diff);
            points2[i].Y() = V.Dot(diff);
        }

        // Compute the minimum area box in 2D.
        box2 = MinBox2<Real>(numPoints, points2, epsilon, queryType, false);
        delete1(points2);

        // Lift the values into 3D.
        mSymBox.Center = origin + box2.Center.X()*U + box2.Center.Y()*V;
        mSymBox.Axis[0] = box2.Axis[0].X()*U + box2.Axis[0].Y()*V;
        mSymBox.Axis[1] = box2.Axis[1].X()*U + box2.Axis[1].Y()*V;
        mSymBox.Axis[2] = W;
        mSymBox.Extent[0] = box2.Extent[0];
        mSymBox.Extent[1] = box2.Extent[1];
        mSymBox.Extent[2] = (Real)0;
        
        return;
    }

    int hullQuantity = 0;//kHull.GetNumSimplices();
	int* tmpIndices;
	const int* hullIndices;
	//bool delTmpIndices = false;
	bool hullFailed = false;

	if (hullQuantity == 0)
    {
		/*
        mSymBox.Center = points[0];
        mSymBox.Axis[0] = Vector3<Real>::UNIT_X;
        mSymBox.Axis[1] = Vector3<Real>::UNIT_Y;
        mSymBox.Axis[2] = Vector3<Real>::UNIT_Z;
        mSymBox.Extent[0] = (Real)0;
        mSymBox.Extent[1] = (Real)0;
        mSymBox.Extent[2] = (Real)0;
        
        mProblem = 1;
        return;
		*/

		hullQuantity = numFaces;
		hullFailed = true;
		//delTmpIndices = true;
		//hullIndices = new int[3*numFaces];
		tmpIndices = new int[3*numFaces];

		for(int f = 0; f < numFaces; f++)
		{
			int v1 = faces[f].X();
			int v2 = faces[f].Y();
			int v3 = faces[f].Z();
			tmpIndices[3*f] = v1;
			tmpIndices[3*f+1] = v2;
			tmpIndices[3*f+2] = v3;
		}

		hullIndices = tmpIndices;
    }
	else
	{
		hullIndices = kHull.GetIndices();
	}
	
    //const int* hullIndices = kHull.GetIndices();
    Real volume, minVolume = Math<Real>::MAX_REAL;

    // Create the unique set of hull vertices to minimize the time spent
    // projecting vertices onto planes of the hull faces.
    std::set<int> uniqueIndices;
	//mHullVerts = new int[3*hullQuantity];
    for (i = 0; i < 3*hullQuantity; ++i)
    {
        uniqueIndices.insert(hullIndices[i]);
		//mHullVerts[i] = hullIndices[i];
    }

    // Use the rotating calipers method on the projection of the hull onto
    // the plane of each face.  Also project the hull onto the normal line
    // of each face.  The minimum area box in the plane and the height on
    // the line produce a containing box.  If its volume is smaller than the
    // current volume, this box is the new candidate for the minimum volume
    // box.  The unique edges are accumulated into a set for use by a later
    // step in the algorithm.
    
    double minScore = 1000;
    Box3<Real> symBox;
    mScores = new double[3*hullQuantity];
    mExtents = new double[3*hullQuantity];
    mAxes = new double[9*hullQuantity];
    mCenters = new double[3*hullQuantity];
    
    mNumBoxes = hullQuantity;
    
    const int* currentHullIndex = hullIndices;
    Real height, minHeight, maxHeight;
    std::set<EdgeKey> edges;
    points2 = new1<Vector2<Real> >(uniqueIndices.size());
    for (i = 0; i < hullQuantity; ++i)
    {
        // Get the triangle.
        int v0 = *currentHullIndex++;
        int v1 = *currentHullIndex++;
        int v2 = *currentHullIndex++;

        // Save the edges for later use.
        edges.insert(EdgeKey(v0, v1));
        edges.insert(EdgeKey(v1, v2));
        edges.insert(EdgeKey(v2, v0));

        // Get 3D coordinate system relative to plane of triangle.
        origin = (points[v0] + points[v1] + points[v2])/(Real)3.0;
        Vector3<Real> edge1 = points[v1] - points[v0];
        Vector3<Real> edge2 = points[v2] - points[v0];
        W = edge2.UnitCross(edge1);  // inner-pointing normal
        if (W == Vector3<Real>::ZERO)
        {
            // The triangle is needle-like, so skip it.
			mScores[3*i] = 100.0;
			mScores[3*i+1] = 100.0;
			mScores[3*i+2] = 100.0;
            continue;
        }

		double pd = W.X() * points[v0].X() + W.Y() * points[v0].Y() + W.Z() * points[v0].Z();
		pd = -1 * pd;
		Plane3<Real> facePlane(W, pd);

		if(hullFailed)
		{
			//double pd = W.X() * points[v0].X() + W.Y() * points[v0].Y() + W.Z() * points[v0].Z();
			//pd = -1 * pd;
			//Plane3<Real> facePlane(W, pd);

			for(int rn = 0; rn < 5; rn++)
			{
				int ver = rand() % numPoints + 1;
				int side = facePlane.WhichSide(points[ver]);
		
				if(side < 0)
				{
					//W.X() = -1 * W.X();
					//W.Y() = -1 * W.Y();
					//W.Z() = -1 * W.Z();
					break;
				}
			}
		}
        else
        {
            int noa = 1;
        }

		/*
		if(hullFailed)
		{
			double pd = W.X() * points[v0].X() + W.Y() * points[v0].Y() + W.Z() * points[v0].Z();
			pd = -1 * pd;
			Plane3<Real> facePlane(W, pd);
			int side = facePlane.WhichSide(massCenter);
		
			if(side < 0)
			{
				W.X() = -1 * W.X();
				W.Y() = -1 * W.Y();
				W.Z() = -1 * W.Z();
			}
		}
		*/

        Vector3<Real>::GenerateComplementBasis(U, V, W);

        // Project points onto plane of triangle, onto normal line of plane.
        // TO DO.  In theory, minHeight should be zero since W points to the
        // interior of the hull.  However, the snap rounding used in the 3D
        // convex hull finder involves loss of precision, which in turn can
        // cause a hull facet to have the wrong ordering (clockwise instead
        // of counterclockwise when viewed from outside the hull).  The
        // height calculations here trap that problem (the incorrectly ordered
        // face will not affect the minimum volume box calculations).
        minHeight = (Real)0;
        maxHeight = (Real)0;
        j = 0;
        std::set<int>::const_iterator iter = uniqueIndices.begin();
        while (iter != uniqueIndices.end())
        {
            int index = *iter++;
            diff = points[index] - origin;
            points2[j].X() = U.Dot(diff);
            points2[j].Y() = V.Dot(diff);
            height = W.Dot(diff);
			//height = facePlane.DistanceTo(points[index]);
            if (height > maxHeight)
            {
                maxHeight = height;
            }
            else if (height < minHeight)
            {
                minHeight = height;
            }

            j++;
        }
        if (-minHeight > maxHeight)
        {
            maxHeight = -minHeight;
        }

        // Compute minimum area box in 2D.
        box2 = MinBox2<Real>((int)uniqueIndices.size(), points2, epsilon,
            queryType, false);

        
        // ****** check symmetry ****** //
        
        
        Vector3<Real> n1 = box2.Axis[0].X()*U + box2.Axis[0].Y()*V;
        Vector3<Real> n2 = box2.Axis[1].X()*U + box2.Axis[1].Y()*V;
        Vector3<Real> n3 = W;
        Vector3<Real> center = origin + box2.Center.X()*U + box2.Center.Y()*V + ((Real)0.5)*maxHeight*W;
        
        
        double* scores = CheckSymPlanes(samples, numSamples, points, numPoints, origFaces, numOrigFaces, n1, center, n2, center, n3, center);
		/*
		double* scores = new double[6];
		scores[3] = 100;
		scores[4] = 100;
		scores[5] = 100;
        */

        /*
        mScores[3*i] = scores[0];
        mScores[3*i+1] = scores[1];
        mScores[3*i+2] = scores[2];
        */
        mScores[3*i] = scores[3];
        mScores[3*i+1] = scores[4];
        mScores[3*i+2] = scores[5];

        delete[] scores;
        

        mExtents[3*i] = box2.Extent[0];
        mExtents[3*i+1] = box2.Extent[1];
        mExtents[3*i+2] = ((Real)0.5)*maxHeight;

        Vector3<Real> a1 = box2.Axis[0].X()*U + box2.Axis[0].Y()*V;
        Vector3<Real> a2 = box2.Axis[1].X()*U + box2.Axis[1].Y()*V;
        Vector3<Real> a3 = W;
        
        mAxes[9*i] = a1.X();
        mAxes[9*i+1] = a1.Y();
        mAxes[9*i+2] = a1.Z();
        mAxes[9*i+3] = a2.X();
        mAxes[9*i+4] = a2.Y();
        mAxes[9*i+5] = a2.Z();
        mAxes[9*i+6] = a3.X();
        mAxes[9*i+7] = a3.Y();
        mAxes[9*i+8] = a3.Z();
        
        
        mCenters[3*i] = center.X();
        mCenters[3*i+1] = center.Y();
        mCenters[3*i+2] = center.Z();
        
        // ****** end check symmetry ****** //
        
  
        
        
        // Update current minimum-volume box (if necessary).
        volume = maxHeight*box2.Extent[0]*box2.Extent[1];
        if (volume < minVolume)
        {
            minVolume = volume;

            // Lift the values into 3D.
            mSymBox.Extent[0] = box2.Extent[0];
            mSymBox.Extent[1] = box2.Extent[1];
            mSymBox.Extent[2] = ((Real)0.5)*maxHeight;
            mSymBox.Axis[0] = box2.Axis[0].X()*U + box2.Axis[0].Y()*V;
            mSymBox.Axis[1] = box2.Axis[1].X()*U + box2.Axis[1].Y()*V;
            mSymBox.Axis[2] = W;
            mSymBox.Center = origin + box2.Center.X()*U + box2.Center.Y()*V
                + mSymBox.Extent[2]*W;
        }
    }

    // The minimum-volume box can also be supported by three mutually
    // orthogonal edges of the convex hull.  For each triple of orthogonal
    // edges, compute the minimum-volume box for that coordinate frame by
    // projecting the points onto the axes of the frame.
    std::set<EdgeKey>::const_iterator e2iter;
    for (e2iter = edges.begin(); e2iter != edges.end(); e2iter++)
    {
        W = points[e2iter->V[1]] - points[e2iter->V[0]];
        W.Normalize();

        std::set<EdgeKey>::const_iterator e1iter = e2iter;
        for (++e1iter; e1iter != edges.end(); e1iter++)
        {
            V = points[e1iter->V[1]] - points[e1iter->V[0]];
            V.Normalize();
            Real dot = V.Dot(W);
            if (Math<Real>::FAbs(dot) > Math<Real>::ZERO_TOLERANCE)
            {
                continue;
            }

            std::set<EdgeKey>::const_iterator e0iter = e1iter;
            for (++e0iter; e0iter != edges.end(); e0iter++)
            {
                U = points[e0iter->V[1]] - points[e0iter->V[0]];
                U.Normalize();
                dot = U.Dot(V);
                if (Math<Real>::FAbs(dot) > Math<Real>::ZERO_TOLERANCE)
                {
                    continue;
                }
                dot = U.Dot(W);
                if (Math<Real>::FAbs(dot) > Math<Real>::ZERO_TOLERANCE)
                {
                    continue;
                }
    
                // The three edges are mutually orthogonal.  Project the
                // hull points onto the lines containing the edges.  Use
                // hull point zero as the origin.
                Real umin = (Real)0, umax = (Real)0;
                Real vmin = (Real)0, vmax = (Real)0;
                Real wmin = (Real)0, wmax = (Real)0;
                origin = points[hullIndices[0]];

                std::set<int>::const_iterator iter = uniqueIndices.begin();
                while (iter != uniqueIndices.end())
                {
                    int index = *iter++;
                    diff = points[index] - origin;

                    Real fU = U.Dot(diff);
                    if (fU < umin)
                    {
                        umin = fU;
                    }
                    else if (fU > umax)
                    {
                        umax = fU;
                    }

                    Real fV = V.Dot(diff);
                    if (fV < vmin)
                    {
                        vmin = fV;
                    }
                    else if (fV > vmax)
                    {
                        vmax = fV;
                    }

                    Real fW = W.Dot(diff);
                    if (fW < wmin)
                    {
                        wmin = fW;
                    }
                    else if (fW > wmax)
                    {
                        wmax = fW;
                    }
                }

                Real uExtent = ((Real)0.5)*(umax - umin);
                Real vExtent = ((Real)0.5)*(vmax - vmin);
                Real wExtent = ((Real)0.5)*(wmax - wmin);

                // Update current minimum-volume box (if necessary).
                volume = uExtent*vExtent*wExtent;
                if (volume < minVolume)
                {
                    minVolume = volume;

                    mSymBox.Extent[0] = uExtent;
                    mSymBox.Extent[1] = vExtent;
                    mSymBox.Extent[2] = wExtent;
                    mSymBox.Axis[0] = U;
                    mSymBox.Axis[1] = V;
                    mSymBox.Axis[2] = W;
                    mSymBox.Center = origin +
                        ((Real)0.5)*(umin+umax)*U +
                        ((Real)0.5)*(vmin+vmax)*V +
                        ((Real)0.5)*(wmin+wmax)*W;
                }
            }
        }
    }

    delete1(points2);

	if(hullFailed)
		delete[] tmpIndices;
}
//----------------------------------------------------------------------------
template <typename Real>
double SymBox3<Real>::CheckSymPlane(const Vector3<Real>* samples, int ns, const Vector3<Real>* points, int np, const Vector3<Real>* faces, int nf, Vector3<Real>& normal, Vector3<Real>& point)
{
    
    double score = 0;
    double a = normal.X();
    double b = normal.Y();
    double c = normal.Z();
    double d = a*point.X() + b*point.Y() + c*point.Z();
    d = -d;
	d = d / sqrt(pow(a,2) + pow(b,2) + pow(c,2));
    
    for(int i = 0; i < ns; i++) {
        // reflect the sample point across the plane
        double temp = 2 * (a * samples[i].X() + b * samples[i].Y() + c * samples[i].Z() + d);
        temp = temp / sqrt(pow(a,2)+pow(b,2)+pow(c,2));
        double x1 = samples[i].X() - temp * a;
        double y1 = samples[i].Y() - temp * b;
        double z1 = samples[i].Z() - temp * c;

        Vector3<double> p(x1, y1, z1);

        double pdist = a*x1 + b*y1 + c*z1 + d;

        double minDist = 1000;
        int minFace = -1;

        for(int j = 0; j < nf; j++) {
            int v1 = faces[j].X();
            int v2 = faces[j].Y();
            int v3 = faces[j].Z();

            Vector3<double> p1(points[v1].X(), points[v1].Y(), points[v1].Z());
            Vector3<double> p2(points[v2].X(), points[v2].Y(), points[v2].Z());
            Vector3<double> p3(points[v3].X(), points[v3].Y(), points[v3].Z());
            Triangle3<double> tri(p1, p2, p3);
            //Triangle3<double> tri(points[v1], points[v2], points[v3]);

            DistPoint3Triangle3<double> dist(p, tri);
            //double distance = dist.Get();
            //For some reason dist.Get() was getting an error of
            //Sqrt(negative number). So, GetSquared allows to check that
            //if tht case is happening
            double distance = dist.GetSquared();
            if (distance < 0.0){
                distance = 0.0;
            }
            if(distance < minDist) {
                minDist = distance;
                minFace = j + 1;
            }

        }

        //out[i] = minDist / pdist; // weight by the distance to the plane
        //out[i] = minDist;
        //out[nS+i] = minFace;
        score = score + minDist;

    }
    
    return sqrt(score);
}
//----------------------------------------------------------------------------
template <typename Real>
double* SymBox3<Real>::CheckSymPlanes(const Vector3<Real>* samples, int ns, const Vector3<Real>* points, int np, const Vector3<Real>* faces, int nf, Vector3<Real>& normal1, Vector3<Real>& point1, Vector3<Real>& normal2, Vector3<Real>& point2, Vector3<Real>& normal3, Vector3<Real>& point3)
{
    
    double score1 = 0;
    double score2 = 0;
    double score3 = 0;
    
    double a1 = normal1.X();
    double b1 = normal1.Y();
    double c1 = normal1.Z();
    double norm1 = sqrt(pow(a1,2) + pow(b1,2) + pow(c1,2));
    double d1 = a1*point1.X() + b1*point1.Y() + c1*point1.Z();
    d1 = -d1;
	d1 = d1 / norm1;
    
    double a2 = normal2.X();
    double b2 = normal2.Y();
    double c2 = normal2.Z();
    double norm2 = sqrt(pow(a2,2) + pow(b2,2) + pow(c2,2));
    double d2 = a2*point2.X() + b2*point2.Y() + c2*point2.Z();
    d2 = -d2;
	d2 = d2 / norm2;
    
    double a3 = normal3.X();
    double b3 = normal3.Y();
    double c3 = normal3.Z();
    double norm3 = sqrt(pow(a3,2) + pow(b3,2) + pow(c3,2));
    double d3 = a3*point3.X() + b3*point3.Y() + c3*point3.Z();
    d3 = -d3;
	d3 = d3 / norm3;
    
    double* dists1 = new double[ns];
    double* dists2 = new double[ns];
    double* dists3 = new double[ns];
    
    for(int i = 0; i < ns; i++) {
        // reflect the sample point across the plane
        double temp1 = 2 * (a1 * samples[i].X() + b1 * samples[i].Y() + c1 * samples[i].Z() + d1);
        temp1 = temp1 / norm1;
        double x1 = samples[i].X() - temp1 * a1;
        double y1 = samples[i].Y() - temp1 * b1;
        double z1 = samples[i].Z() - temp1 * c1;

        Vector3<double> p1(x1, y1, z1);
        
        double temp2 = 2 * (a2 * samples[i].X() + b2 * samples[i].Y() + c2 * samples[i].Z() + d2);
        temp2 = temp2 / norm2;
        double x2 = samples[i].X() - temp2 * a2;
        double y2 = samples[i].Y() - temp2 * b2;
        double z2 = samples[i].Z() - temp2 * c2;

        Vector3<double> p2(x2, y2, z2);
        
        double temp3 = 2 * (a3 * samples[i].X() + b3 * samples[i].Y() + c3 * samples[i].Z() + d3);
        temp3 = temp3 / norm3;
        double x3 = samples[i].X() - temp3 * a3;
        double y3 = samples[i].Y() - temp3 * b3;
        double z3 = samples[i].Z() - temp3 * c3;

        Vector3<double> p3(x3, y3, z3);

        //double pdist = a*x1 + b*y1 + c*z1 + d;

        double minDist1 = 1000;
        double minDist2 = 1000;
        double minDist3 = 1000;
        
        //int minFace = -1;

        for(int j = 0; j < nf; j++) {
            int v1 = faces[j].X();
            int v2 = faces[j].Y();
            int v3 = faces[j].Z();

            Vector3<double> tp1(points[v1].X(), points[v1].Y(), points[v1].Z());
            Vector3<double> tp2(points[v2].X(), points[v2].Y(), points[v2].Z());
            Vector3<double> tp3(points[v3].X(), points[v3].Y(), points[v3].Z());
            Triangle3<double> tri(tp1, tp2, tp3);
            //Triangle3<double> tri(points[v1], points[v2], points[v3]);

            DistPoint3Triangle3<double> dist1(p1, tri);
            DistPoint3Triangle3<double> dist2(p2, tri);
            DistPoint3Triangle3<double> dist3(p3, tri);
            
            //double distance = dist.Get();
            //For some reason dist.Get() was getting an error of
            //Sqrt(negative number). So, GetSquared allows to check that
            //if tht case is happening
            
            double distance1 = dist1.GetSquared();
            double distance2 = dist2.GetSquared();
            double distance3 = dist3.GetSquared();
            
            if (distance1 < 0.0){
                distance1 = 0.0;
            }
            if (distance2 < 0.0){
                distance2 = 0.0;
            }
            if (distance3 < 0.0){
                distance3 = 0.0;
            }
            
            if(distance1 < minDist1) {
                minDist1 = distance1;
            }
            if(distance2 < minDist2) {
                minDist2 = distance2;
            }
            if(distance3 < minDist3) {
                minDist3 = distance3;
            }

        }
        
        score1 = score1 + minDist1;
        score2 = score2 + minDist2;
        score3 = score3 + minDist3;
        
        dists1[i] = minDist1;
        dists2[i] = minDist2;
        dists3[i] = minDist3;
    }
    
    double* scores = new double[6];
    scores[0] = score1/ns;
    scores[1] = score2/ns;
    scores[2] = score3/ns;
    
    qsort(dists1, ns, sizeof(double), compare);
    qsort(dists2, ns, sizeof(double), compare);
    qsort(dists3, ns, sizeof(double), compare);
    
    int thr = 0.9 * ns;
    scores[3] = dists1[thr];
    scores[4] = dists2[thr];
    scores[5] = dists3[thr];

	delete[] dists1;
	delete[] dists2;
	delete[] dists3;

    return scores;
}
//----------------------------------------------------------------------------
template <typename Real>
SymBox3<Real>::operator Box3<Real> () const
{
    return mSymBox;
}
//----------------------------------------------------------------------------
template <typename Real>
double* SymBox3<Real>::getScores()
{
    return mScores;
}
template <typename Real>
double* SymBox3<Real>::getExtents()
{
    return mExtents;
}
template <typename Real>
double* SymBox3<Real>::getAxes()
{
    return mAxes;
}
template <typename Real>
double* SymBox3<Real>::getCenters()
{
    return mCenters;
}
template <typename Real>
int SymBox3<Real>::getNumBoxes()
{
    return mNumBoxes;
}
template <typename Real>
int SymBox3<Real>::getProblem()
{
    return mProblem;
}
/*
template <typename Real>
int* SymBox3<Real>::getHullVerts()
{
    return mHullVerts;
}
*/
/*
template <typename Real>
double* SymBox3<Real>::getNormals()
{
    return mNormals;
}
template <typename Real>
double* SymBox3<Real>::getPoints()
{
    return mPoints;
}
template <typename Real>
int SymBox3<Real>::getNumOfPlanes()
{
    return mNumPlanes;
}
*/

template <typename Real>
void SymBox3<Real>::freeFields()
{
    delete[] mScores;
    //delete[] mNormals;
    //delete[] mPoints;
    delete[] mExtents;
    delete[] mAxes;
    delete[] mCenters;

	//delete[] mHullVerts;
}
//----------------------------------------------------------------------------
// Explicit instantiation.
//----------------------------------------------------------------------------
template WM5_MATHEMATICS_ITEM
class SymBox3<float>;

template WM5_MATHEMATICS_ITEM
class SymBox3<double>;
//----------------------------------------------------------------------------
}
