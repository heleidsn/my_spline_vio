/**
 * This file is part of DSO.
 *
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "map"
#include "util/IndexThreadReduce.h"
#include "util/NumType.h"
#include "vector"
#include <math.h>

namespace dso {

class PointFrameResidual;   //!< 在点到帧的重投影过程中产生的残差
class CalibHessian;         //!< 相机内参的Hessian矩阵
class FrameHessian;         //!< 帧的Hessian矩阵
class PointHessian;         //!< 点的Hessian矩阵

class EFResidual;           //!< 残差的能量函数
class EFPoint;              //!< 点的能量函数
class EFFrame;              //!< 帧的能量函数
class EnergyFunctional;     //!< 总领全局 协调各方
class AccumulatedTopHessian;            
class AccumulatedTopHessianSSE;         
class AccumulatedSCHessian;             
class AccumulatedSCHessianSSE;

extern bool EFAdjointsValid;
extern bool EFIndicesValid;
extern bool EFDeltaValid;

class EnergyFunctional {  //!< 总领全局 协调各方
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  friend class EFFrame;
  friend class EFPoint;
  friend class EFResidual;
  friend class AccumulatedTopHessian;
  friend class AccumulatedTopHessianSSE;
  friend class AccumulatedSCHessian;
  friend class AccumulatedSCHessianSSE;

  EnergyFunctional();
  ~EnergyFunctional();

  EFResidual *insertResidual(PointFrameResidual *r);
  EFFrame *insertFrame(FrameHessian *fh, CalibHessian *HCalib);
  EFPoint *insertPoint(PointHessian *ph);

  void dropResidual(EFResidual *r);
  void marginalizeFrame(EFFrame *fh, CalibHessian *HCalib);
  void removePoint(EFPoint *ph);

  void marginalizePointsF();
  void dropPointsF();
  void solveSystemF(int iteration, double lambda, CalibHessian *HCalib);
  double calcMEnergyF();
  double calcLEnergyF_MT();

  void makeIDX();

  void setDeltaF(CalibHessian *HCalib);

  void setAdjointsF(CalibHessian *HCalib);

  std::vector<EFFrame *> frames;
  int nPoints, nFrames, nResiduals;

  int resInA, resInL, resInM;
  VecX lastX;
  std::vector<VecX> lastNullspaces_forLogging;
  std::vector<VecX> lastNullspaces_pose;
  std::vector<VecX> lastNullspaces_scale;
  std::vector<VecX> lastNullspaces_affA;
  std::vector<VecX> lastNullspaces_affB;

  IndexThreadReduce<Vec10> *red;

  std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>,
           Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>>
      connectivityMap;

  // ToDo: move to private
  void getImuHessian(MatXX &H, VecX &b, MatXX &J_cst, VecX &r_cst,
                     CalibHessian *HCalib, std::vector<bool> &is_spline_valid,
                     bool print = false);

private:
  VecX getStitchedDeltaF() const;

  void resubstituteF_MT(VecX x, CalibHessian *HCalib, bool MT);
  void resubstituteFPt(const VecCf &xc, Mat18f *xAd, int min, int max,
                       Vec10 *stats, int tid);

  void accumulateAF_MT(MatXX &H, VecX &b, bool MT);
  void accumulateLF_MT(MatXX &H, VecX &b, bool MT);
  void accumulateSCF_MT(MatXX &H, VecX &b, bool MT);

  void expandHbtoFitImu(MatXX &H, VecX &b);

  void getImuHessianCurrentFrame(int fi, CalibHessian *HCalib, MatXX &H,
                                 VecX &b, bool &spline_valid, MatXX &J_vr,
                                 VecX &r_vr, bool print);

  void calcLEnergyPt(int min, int max, Vec10 *stats, int tid);

  void orthogonalize(VecX *b, MatXX *H);
  Mat18f *adHTdeltaF;

  Mat88 *adHost;
  Mat88 *adTarget;

  Mat88f *adHostF;
  Mat88f *adTargetF;

  VecC cPrior;
  VecCf cDeltaF;
  VecCf cPriorF;

  MatXX HM;
  VecX bM;

  MatXX HM_bias;
  VecX bM_bias;

  MatXX HM_imu;
  VecX bM_imu;

  AccumulatedTopHessianSSE *accSSE_top_L;
  AccumulatedTopHessianSSE *accSSE_top_A;

  AccumulatedSCHessianSSE *accSSE_bot;

  std::vector<EFPoint *> allPoints;
  std::vector<EFPoint *> allPointsToMarg;

  float currentLambda;
};
} // namespace dso
