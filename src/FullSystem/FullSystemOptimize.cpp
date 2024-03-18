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

#include "FullSystem.h"

#include "IOWrapper/ImageDisplay.h"
#include "ResidualProjections.h"
#include "stdio.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include <cmath>

#include <algorithm>

namespace dso {

void FullSystem::linearizeAll_Reductor(
    bool fixLinearization, std::vector<PointFrameResidual *> *toRemove, int min,
    int max, Vec10 *stats, int tid) {
  for (int k = min; k < max; k++) {
    PointFrameResidual *r = activeResiduals[k];
    (*stats)[0] += r->linearize(&HCalib);

    if (fixLinearization) {
      r->applyRes(true);

      if (r->efResidual->isActive()) {
        if (r->isNew) {
          PointHessian *p = r->point;
          Vec3f ptp_inf =
              r->host->targetPrecalc[r->target->idx].PRE_KRKiTll *
              Vec3f(p->u, p->v, 1); // projected point assuming infinite depth.
          Vec3f ptp = ptp_inf +
                      r->host->targetPrecalc[r->target->idx].PRE_KtTll *
                          p->idepth_scaled; // projected point with real depth.
          float relBS = 0.01 * ((ptp_inf.head<2>() / ptp_inf[2]) -
                                (ptp.head<2>() / ptp[2]))
                                   .norm(); // 0.01 = one pixel.

          if (relBS > p->maxRelBaseline)
            p->maxRelBaseline = relBS;

          p->numGoodResiduals++;
        }
      } else {
        toRemove[tid].push_back(activeResiduals[k]);
      }
    }
  }
}

void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max,
                                   Vec10 *stats, int tid) {
  for (int k = min; k < max; k++)
    activeResiduals[k]->applyRes(true);
}
void FullSystem::setNewFrameEnergyTH() {

  // collect all residuals and make decision on TH.
  allResVec.clear();
  allResVec.reserve(activeResiduals.size() * 2);
  FrameHessian *newFrame = frameHessians.back();

  for (PointFrameResidual *r : activeResiduals)
    if (r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame) {
      allResVec.push_back(r->state_NewEnergyWithOutlier);
    }

  if (allResVec.size() == 0) {
    newFrame->frameEnergyTH = 12 * 12 * patternNum;
    return; // should never happen, but lets make sure.
  }

  int nthIdx = setting_frameEnergyTHN * allResVec.size();

  assert(nthIdx < (int)allResVec.size());
  assert(setting_frameEnergyTHN < 1);

  std::nth_element(allResVec.begin(), allResVec.begin() + nthIdx,
                   allResVec.end());
  float nthElement = sqrtf(allResVec[nthIdx]);

  newFrame->frameEnergyTH = nthElement * setting_frameEnergyTHFacMedian;
  newFrame->frameEnergyTH =
      26.0f * setting_frameEnergyTHConstWeight +
      newFrame->frameEnergyTH * (1 - setting_frameEnergyTHConstWeight);
  newFrame->frameEnergyTH = newFrame->frameEnergyTH * newFrame->frameEnergyTH;
  newFrame->frameEnergyTH *=
      setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;

  //
  //	int good=0,bad=0;
  //	for(float f : allResVec) if(f<newFrame->frameEnergyTH) good++; else
  // bad++; 	printf("EnergyTH: mean %f, median %f, result %f (in %d, out %d)!
  // \n", 			meanElement, nthElement,
  // sqrtf(newFrame->frameEnergyTH), good, bad);
}
Vec3 FullSystem::linearizeAll(bool fixLinearization) {
  double lastEnergyP = 0;
  double lastEnergyR = 0;
  double num = 0;

  std::vector<PointFrameResidual *> toRemove[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; i++)
    toRemove[i].clear();

  if (multiThreading) {
    treadReduce.reduce(boost::bind(&FullSystem::linearizeAll_Reductor, this,
                                   fixLinearization, toRemove, _1, _2, _3, _4),
                       0, activeResiduals.size(), 0);
    lastEnergyP = treadReduce.stats[0];
  } else {
    Vec10 stats;
    linearizeAll_Reductor(fixLinearization, toRemove, 0, activeResiduals.size(),
                          &stats, 0);
    lastEnergyP = stats[0];
  }

  setNewFrameEnergyTH();

  if (fixLinearization) {

    for (PointFrameResidual *r : activeResiduals) {
      PointHessian *ph = r->point;
      if (ph->lastResiduals[0].first == r)
        ph->lastResiduals[0].second = r->state_state;
      else if (ph->lastResiduals[1].first == r)
        ph->lastResiduals[1].second = r->state_state;
    }

    int nResRemoved = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
      for (PointFrameResidual *r : toRemove[i]) {
        PointHessian *ph = r->point;

        if (ph->lastResiduals[0].first == r)
          ph->lastResiduals[0].first = 0;
        else if (ph->lastResiduals[1].first == r)
          ph->lastResiduals[1].first = 0;

        for (unsigned int k = 0; k < ph->residuals.size(); k++)
          if (ph->residuals[k] == r) {
            ef->dropResidual(r->efResidual);
            deleteOut<PointFrameResidual>(ph->residuals, k);
            nResRemoved++;
            break;
          }
      }
    }
    // printf("FINAL LINEARIZATION: removed %d / %d residuals!\n", nResRemoved,
    // (int)activeResiduals.size());
  }

  return Vec3(lastEnergyP, lastEnergyR, num);
}

// applies step to linearization point.
bool FullSystem::doStepFromBackup(float stepfacC, float stepfacT,
                                  float stepfacR, float stepfacA,
                                  float stepfacD) {
  //	float meanStepC=0,meanStepP=0,meanStepD=0;
  //	meanStepC += HCalib.step.norm();

  Vec10 pstepfac;
  pstepfac.segment<3>(0).setConstant(stepfacT);
  pstepfac.segment<3>(3).setConstant(stepfacR);
  pstepfac.segment<4>(6).setConstant(stepfacA);

  float sumA = 0, sumB = 0, sumT = 0, sumR = 0, sumID = 0, numID = 0;

  float sumNID = 0;

  HCalib.setValue(HCalib.value_backup + stepfacC * HCalib.step);
  for (FrameHessian *fh : frameHessians) {
    fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
    sumA += fh->step[6] * fh->step[6];
    sumB += fh->step[7] * fh->step[7];
    sumT += fh->step.segment<3>(0).squaredNorm();
    sumR += fh->step.segment<3>(3).squaredNorm();

    for (PointHessian *ph : fh->pointHessians) {
      ph->setIdepth(ph->idepth_backup + stepfacD * ph->step);
      sumID += ph->step * ph->step;
      sumNID += fabsf(ph->idepth_backup);
      numID++;

      ph->setIdepthZero(ph->idepth_backup + stepfacD * ph->step);
    }
  }

  // IMU and scale update
  if (setting_enable_imu && HCalib.imu_initialized) {
    Vec3 stepfacSg;
    stepfacSg.setConstant(stepfacR);
    HCalib.setSg(HCalib.sg_backup + stepfacSg.cwiseProduct(HCalib.sg_step));

    Vec21 pstepfac_imu;
    pstepfac_imu.segment<6>(0) = pstepfac.segment<6>(0);
    pstepfac_imu.segment<3>(6).setConstant(stepfacR);
    pstepfac_imu.segment<6>(9) = pstepfac.segment<6>(0);
    pstepfac_imu.segment<6>(15) = pstepfac.segment<6>(0);
    for (FrameHessian *fh : frameHessians) {
      fh->setImuState(fh->state_imu_backup +
                      pstepfac_imu.cwiseProduct(fh->step_imu));
    }
  }

  sumA /= frameHessians.size();
  sumB /= frameHessians.size();
  sumR /= frameHessians.size();
  sumT /= frameHessians.size();
  sumID /= numID;
  sumNID /= numID;

  if (!setting_debugout_runquiet)
    printf("STEPS: A %.1f; B %.1f; R %.1f; T %.1f. \t",
           sqrtf(sumA) / (0.0005 * setting_thOptIterations),
           sqrtf(sumB) / (0.00005 * setting_thOptIterations),
           sqrtf(sumR) / (0.00005 * setting_thOptIterations),
           sqrtf(sumT) * sumNID / (0.00005 * setting_thOptIterations));

  EFDeltaValid = false;
  setPrecalcValues();

  return sqrtf(sumA) < 0.0005 * setting_thOptIterations &&
         sqrtf(sumB) < 0.00005 * setting_thOptIterations &&
         sqrtf(sumR) < 0.00005 * setting_thOptIterations &&
         sqrtf(sumT) * sumNID < 0.00005 * setting_thOptIterations;
  //
  //	printf("mean steps: %f %f %f!\n",
  //			meanStepC, meanStepP, meanStepD);
}

// sets linearization point.
void FullSystem::backupState(bool backupLastStep) {
  HCalib.value_backup = HCalib.value;
  HCalib.sg_backup = HCalib.sg;
  for (FrameHessian *fh : frameHessians) {
    fh->state_backup = fh->get_state();
    fh->state_imu_backup = fh->getImuState();
    for (PointHessian *ph : fh->pointHessians)
      ph->idepth_backup = ph->idepth;
  }
}

// sets linearization point.
void FullSystem::loadSateBackup() {
  HCalib.setValue(HCalib.value_backup);
  HCalib.setSg(HCalib.sg_backup);
  for (FrameHessian *fh : frameHessians) {
    fh->setState(fh->state_backup);
    fh->setImuState(fh->state_imu_backup);
    for (PointHessian *ph : fh->pointHessians) {
      ph->setIdepth(ph->idepth_backup);

      ph->setIdepthZero(ph->idepth_backup);
    }
  }

  EFDeltaValid = false;
  setPrecalcValues();
}

double FullSystem::calcMEnergy() {
  if (setting_forceAceptStep)
    return 0;
  // calculate (x-x0)^T * [2b + H * (x-x0)] for everything saved in L.
  // ef->makeIDX();
  // ef->setDeltaF(&HCalib);
  return ef->calcMEnergyF();
}

void FullSystem::printOptRes(const Vec3 &res, double resL, double resM,
                             double resPrior, double LExact, float a, float b) {
  printf("A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n", res[0],
         sqrtf((float)(res[0] / (patternNum * ef->resInA))), ef->resInA,
         ef->resInM, a, b);
}
//! ================＝＝＝＝＝＝＝=======重点＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
float FullSystem::optimize(int mnumOptIts) {   //! ref:https://www.cnblogs.com/JingeTU/p/8395046.html

  if (frameHessians.size() < 2)
    return 0;
  if (frameHessians.size() < 3)
    mnumOptIts = 20;
  if (frameHessians.size() < 4)
    mnumOptIts = 15;

  // get statistics and active residuals.

  activeResiduals.clear();   //!< 清空activeResiduals
  int numPoints = 0;
  int numLRes = 0;
  for (FrameHessian *fh : frameHessians)                //!< 遍历所有帧
    for (PointHessian *ph : fh->pointHessians) {        //!< 遍历所有点
      for (PointFrameResidual *r : ph->residuals) {  //!< 遍历所有投影残差
        if (!r->efResidual->isLinearized) {         
          activeResiduals.push_back(r);             //! 将没有被线性化（或者没有求导）的点投影残差r加入activeResiduals  有大概10000个activeResiduals 其实这个里面的全部都线性化了
          r->resetOOB();
        } else
          numLRes++;
      }
      numPoints++;
    }

  if (!setting_debugout_runquiet)
    printf("OPTIMIZE %d pts, %d active res, %d lin res!\n", ef->nPoints,
           (int)activeResiduals.size(), numLRes); //! 输出优化点数，优化残差数，线性化残差数 一般好像是2000个左右

  Vec3 lastEnergy = linearizeAll(false);  //! 计算相关导数
  double lastEnergyL = calcLEnergy();     //! 计算优化前的光度能量
  double lastEnergyM = calcMEnergy();     //! 计算优化前的运动能量
  //! ---------------迭代之前先求偏导------------------ 调用applyRes_Reductor函数，这个函数对好点做一次中间量的计算，输入数据可看成是linearize返回的偏导值
  if (multiThreading)
    treadReduce.reduce(
        boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4),
        0, activeResiduals.size(), 50);
  else
    applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);

  if (!setting_debugout_runquiet) {
    printf("Initial Error       \t");
    printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0,
                frameHessians.back()->aff_g2l().a,
                frameHessians.back()->aff_g2l().b);
  }

  debugPlotTracking();
  //! ==================迭代优化===============
  double lambda = 1e-1;   //! lambda刚开始为0.1，此后每次变为原来的1/4
  float stepsize = 1;
  VecX previousX = VecX::Constant(CPARS + 8 * frameHessians.size(), NAN);
  for (int iteration = 0; iteration < mnumOptIts; iteration++) {  //! ==================迭代优化===============
    // solve!
    backupState(iteration != 0);  //* 保存当前的状态，这是为了在优化结果不好的情况下可以回退,backupState函数实现。
    // solveSystemNew(0);
    solveSystem(iteration, lambda);  //! Step1:===============得到优化变量step =====================
    double incDirChange = (1e-20 + previousX.dot(ef->lastX)) /
                          (1e-20 + previousX.norm() * ef->lastX.norm());
    previousX = ef->lastX;

    bool canbreak =  //! 但是为什么目前来看这stepsize都是1？ todo: 需要调试看一下
        doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);  //! Step2: =======让优化变量生效============= applies step to linearization point.

    // eval new energy!  //! ==============计算新的能量====================== 用上面计算得到的新状态值计算一次新的残差以及偏导数等
    Vec3 newEnergy = linearizeAll(false);  // 这里的false是指不需要进行修复
    double newEnergyL = calcLEnergy();
    double newEnergyM = calcMEnergy();

    if (!setting_debugout_runquiet) {  //! ==============输出更新结果======================
      printf("%s %d (L %.2f, dir %.2f, ss %.1f): \t",
             (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
              lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM)  //! 如果新的能量小于上一次的能量，就是Accept，否则reject
                 ? "ACCEPT"
                 : "REJECT",
             iteration, log10(lambda), incDirChange, stepsize);
      printOptRes(newEnergy, newEnergyL, newEnergyM, 0, 0,
                  frameHessians.back()->aff_g2l().a,
                  frameHessians.back()->aff_g2l().b);
    }
    //! ==============判断新的能量和旧的能量之间的大小关系====================== 如果新残差值小于原来残差值，那么就再调用一次调用applyRes_Reductor函数，接着调整lambda值，否则就回滚到上一次的状态
    if (setting_forceAceptStep ||
        (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
         lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM)) {
      //! 求偏导数  迭代过程中求 随后继续迭代这样
      if (multiThreading)
        treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this,
                                       true, _1, _2, _3, _4),
                           0, activeResiduals.size(), 50);
      else
        applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);

      lastEnergy = newEnergy;
      lastEnergyL = newEnergyL;
      lastEnergyM = newEnergyM;

      lambda *= 0.25;  //! lambda进行改变
    } else {  // 利用之前保存的状态进行回滚
      loadSateBackup();
      lastEnergy = linearizeAll(false);  // 效果是将优化之后成为 outlier 的 residual 剔除，剩下正常的 residual 调用一次
      lastEnergyL = calcLEnergy();
      lastEnergyM = calcMEnergy();
      lambda *= 1e2;
    }

    if (canbreak && iteration >= setting_minOptIterations)
      break;
  }

  Vec10 newStateZero = Vec10::Zero();
  newStateZero.segment<2>(6) = frameHessians.back()->get_state().segment<2>(6);

  frameHessians.back()->setEvalPT(frameHessians.back()->PRE_camToWorld,
                                  newStateZero);
  EFDeltaValid = false;
  EFAdjointsValid = false;
  ef->setAdjointsF(&HCalib);
  setPrecalcValues();

  lastEnergy = linearizeAll(true);  // 在跳出循环体之后调用一次 FullSystem::linearizeAll(true)，效果是将优化之后成为 outlier 的 residual 剔除，剩下正常的 residual 调用一次

  if (!std::isfinite((double)lastEnergy[0]) ||
      !std::isfinite((double)lastEnergy[1]) ||
      !std::isfinite((double)lastEnergy[2])) {
    printf("KF Tracking failed: LOST!\n");
    isLost = true;
  }

  statistics_lastFineTrackRMSE =
      sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));

  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    for (FrameHessian *fh : frameHessians) {
      fh->shell->scale = HCalib.getScaleScaled();
      fh->shell->camToWorld = fh->PRE_camToWorld;
      fh->shell->aff_g2l = fh->aff_g2l();
    }
  }

  debugPlotTracking();

  if (setting_enable_imu) {
    if (HCalib.imu_initialized) {
      frameHessians.back()->updateVel(
          frameHessians[frameHessians.size() - 2]->shell);
      frameHessians.back()->setImuStateZero(&HCalib);

      if (setting_print_imu) {
        MatXX H_tmp, J_tmp;
        VecX b_tmp, r_tmp;
        std::vector<bool> v_tmp;
        ef->getImuHessian(H_tmp, b_tmp, J_tmp, r_tmp, &HCalib, v_tmp, true);  //! 有一些输出，应该重点是改这里
      }
    }

    if (!HCalib.scale_trapped) {
      HCalib.tryTrapScale();
      if (HCalib.scale_trapped) {
        for (FrameHessian *fh : frameHessians) {
          fh->setImuStateZero(&HCalib);
        }
      }
    }
  }

  return sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));
}

void FullSystem::solveSystem(int iteration, double lambda) {
  ef->lastNullspaces_forLogging =
      getNullspaces(ef->lastNullspaces_pose, ef->lastNullspaces_scale,
                    ef->lastNullspaces_affA, ef->lastNullspaces_affB);

  ef->solveSystemF(iteration, lambda, &HCalib);
}

double FullSystem::calcLEnergy() {
  if (setting_forceAceptStep)
    return 0;

  double Ef = ef->calcLEnergyF_MT();
  return Ef;
}

void FullSystem::removeOutliers() {
  int numPointsDropped = 0;
  for (FrameHessian *fh : frameHessians) {
    for (unsigned int i = 0; i < fh->pointHessians.size(); i++) {
      PointHessian *ph = fh->pointHessians[i];
      if (ph == 0)
        continue;

      if (ph->residuals.size() == 0) {
        fh->pointHessiansOut.push_back(ph);
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        fh->pointHessians[i] = fh->pointHessians.back();
        fh->pointHessians.pop_back();
        i--;
        numPointsDropped++;
      }
    }
  }
  ef->dropPointsF();
}

std::vector<VecX> FullSystem::getNullspaces(
    std::vector<VecX> &nullspaces_pose, std::vector<VecX> &nullspaces_scale,
    std::vector<VecX> &nullspaces_affA, std::vector<VecX> &nullspaces_affB) {
  nullspaces_pose.clear();
  nullspaces_scale.clear();
  nullspaces_affA.clear();
  nullspaces_affB.clear();

  int n = CPARS + frameHessians.size() * 8;
  std::vector<VecX> nullspaces_x0_pre;
  for (int i = 0; i < 6; i++) {
    VecX nullspace_x0(n);
    nullspace_x0.setZero();
    for (FrameHessian *fh : frameHessians) {
      nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_pose.col(i);
      nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
      nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
    }
    nullspaces_x0_pre.push_back(nullspace_x0);
    nullspaces_pose.push_back(nullspace_x0);
  }
  for (int i = 0; i < 2; i++) {
    VecX nullspace_x0(n);
    nullspace_x0.setZero();
    for (FrameHessian *fh : frameHessians) {
      nullspace_x0.segment<2>(CPARS + fh->idx * 8 + 6) =
          fh->nullspaces_affine.col(i).head<2>();
      nullspace_x0[CPARS + fh->idx * 8 + 6] *= SCALE_A_INVERSE;
      nullspace_x0[CPARS + fh->idx * 8 + 7] *= SCALE_B_INVERSE;
    }
    nullspaces_x0_pre.push_back(nullspace_x0);
    if (i == 0)
      nullspaces_affA.push_back(nullspace_x0);
    if (i == 1)
      nullspaces_affB.push_back(nullspace_x0);
  }

  VecX nullspace_x0(n);
  nullspace_x0.setZero();
  for (FrameHessian *fh : frameHessians) {
    nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_scale;
    nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
    nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
  }
  nullspaces_x0_pre.push_back(nullspace_x0);
  nullspaces_scale.push_back(nullspace_x0);

  return nullspaces_x0_pre;
}

} // namespace dso
