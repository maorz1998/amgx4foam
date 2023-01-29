/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019-2020 OpenCFD Ltd.
    Copyright (C) 2019 Simone Bna
    Copyright (C) 2020 Stefano Zampini
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "lduMatrix.H"
#include "error.H"
#include "petscUtils.H"
#include "petscLinearSolverContext.H"

// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

Foam::scalar Foam::gAverage(Vec input)
{
    PetscInt len;
    VecGetSize(input, &len);

    if (len)
    {
        PetscScalar val;
        VecSum(input, &val);

        return val/len;
    }

    WarningInFunction
        << "Empty PETSc Vec, returning zero" << endl;

    return 0;
}


Foam::scalar Foam::gSum(Vec input)
{
    PetscScalar val;
    VecSum(input, &val);

    return val;
}


Foam::scalar Foam::PetscUtils::normFactor
(
    Vec AdotPsi,
    Vec psi,
    Vec source,
    Vec ArowsSum
)
{
    // Equivalent to the OpenFOAM normFactor function
    //
    // stabilise
    // (
    //   gSum(cmptMag(Apsi - tmpField) + cmptMag(matrix_.source() - tmpField)),
    //   SolverPerformance<Type>::small_
    // )

    PetscScalar avgPsi;
    {
        VecSum(psi, &avgPsi);

        PetscInt len;
        VecGetSize(psi, &len);
        avgPsi /= len;
    }

    const PetscScalar* ArowsSumVecValues;
    VecGetArrayRead(ArowsSum, &ArowsSumVecValues);

    const PetscScalar* AdotPsiValues;
    VecGetArrayRead(AdotPsi, &AdotPsiValues);

    const PetscScalar* sourceValues;
    VecGetArrayRead(source, &sourceValues);

    scalar normFactor{0};

    PetscInt len;
    VecGetLocalSize(psi, &len);

    for (PetscInt i=0; i < len; ++i)
    {
        const PetscScalar psiRow = (ArowsSumVecValues[i] * avgPsi);

        normFactor +=
        (
            Foam::mag(AdotPsiValues[i] - psiRow)
          + Foam::mag(sourceValues[i] - psiRow)
        );
    }

    // Restore
    VecRestoreArrayRead(ArowsSum, &ArowsSumVecValues);
    VecRestoreArrayRead(AdotPsi, &AdotPsiValues);
    VecRestoreArrayRead(source, &sourceValues);

    return stabilise
    (
        returnReduce(normFactor, sumOp<scalar>()),
        SolverPerformance<scalar>::small_
    );
}


PetscErrorCode Foam::PetscUtils::foamKSPMonitorFoam
(
    KSP ksp,
    PetscInt it,
    PetscReal rnorm,
    void *cctx
)
{
    PetscErrorCode ierr;
    PetscViewer viewer;
    PetscInt tablevel;
    const char *prefix;
    PetscReal fnorm;
    KSPNormType ntype;

    PetscFunctionBeginUser;
    auto* ctx = static_cast<petscLinearSolverContext*>(cctx);

    // compute L1 norm and rescale by normFactor
    ierr = KSPBuildResidual(ksp, ctx->res_l1_w[0], ctx->res_l1_w[1], &ctx->res_l1_w[1]);CHKERRQ(ierr);
    ierr = VecNorm(ctx->res_l1_w[1], NORM_1, &fnorm);CHKERRQ(ierr);
    fnorm /= ctx->normFactor;
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ksp), &viewer);CHKERRQ(ierr);
    ierr = PetscObjectGetTabLevel((PetscObject)ksp, &tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIAddTab(viewer, tablevel);CHKERRQ(ierr);
    ierr = KSPGetOptionsPrefix(ksp, &prefix);CHKERRQ(ierr);
    if (it == 0)
    {
       ierr = PetscViewerASCIIPrintf(viewer, "  Residual norms for %s solve.\n", prefix);CHKERRQ(ierr);
    }
    ierr = KSPGetNormType(ksp, &ntype);CHKERRQ(ierr);
    if (ntype != KSP_NORM_NONE) // Print both norms, KSP built-in and OpenFOAM built-in
    {
       char normtype[256];
       PetscErrorCode (*converge)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*);

       ierr = PetscStrncpy(normtype,KSPNormTypes[ntype],sizeof(normtype));CHKERRQ(ierr);
       ierr = PetscStrtolower(normtype);CHKERRQ(ierr);
       ierr = KSPGetConvergenceTest(ksp, &converge, NULL, NULL);CHKERRQ(ierr);
       if (converge == Foam::PetscUtils::foamKSPConverge) // we are using foam convergence testing, list foam norm first
       {
          ierr = PetscViewerASCIIPrintf(viewer, "%3D KSP Residual foam norm %14.12e (PETSc %s norm %14.12e)\n", it, (double)fnorm, normtype, (double)rnorm);CHKERRQ(ierr);
       }
       else // we are using KSP default convergence testing, list KSP norm first
       {
          ierr = PetscViewerASCIIPrintf(viewer, "%3D KSP Residual %s norm %14.12e (foam norm %14.12e)\n", it, normtype, (double)rnorm, (double)fnorm);CHKERRQ(ierr);
       }
    }
    else // KSP has no norm, list foam norm only
    {
       ierr = PetscViewerASCIIPrintf(viewer, "%3D KSP Residual foam norm %14.12e\n", it, (double)fnorm);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISubtractTab(viewer, tablevel);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


PetscErrorCode Foam::PetscUtils::foamKSPMonitorRecordInit
(
    KSP ksp,
    PetscInt it,
    PetscReal rnorm,
    void *cctx
)
{
    PetscFunctionBeginUser;
    if (it) PetscFunctionReturn(0);

    auto* ctx = static_cast<petscLinearSolverContext*>(cctx);
    solverPerformance& solverPerf = ctx->performance;

    solverPerf.initialResidual() = rnorm;
    PetscFunctionReturn(0);
}


PetscErrorCode Foam::PetscUtils::foamKSPConverge
(
    KSP ksp,
    PetscInt it,
    PetscReal rnorm,
    KSPConvergedReason* reason,
    void* cctx
)
{
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    //
    // Equivalent to the OpenFOAM checkConvergence function
    //
    auto* ctx = static_cast<petscLinearSolverContext*>(cctx);
    solverPerformance& solverPerf = ctx->performance;

    PetscReal rtol;
    PetscReal abstol;
    PetscReal divtol;
    PetscInt maxits;
    ierr = KSPGetTolerances(ksp, &rtol, &abstol, &divtol, &maxits);CHKERRQ(ierr);

    // compute L1 norm of residual (PETSc always uses L2)
    // assumes normFactor have been precomputed before solving the linear system
    // When using CG, this is actually a copy of the residual vector
    // stored inside the PETSc class (from PETSc version 3.14 on).
    // With GMRES instead, this call is more expensive
    // since we first need to generate the solution
    ierr = KSPBuildResidual(ksp, ctx->res_l1_w[0], ctx->res_l1_w[1], &ctx->res_l1_w[1]);CHKERRQ(ierr);
    ierr = VecNorm(ctx->res_l1_w[1], NORM_1, &rnorm);CHKERRQ(ierr);

    // rescale by the normFactor
    PetscReal residual = rnorm / ctx->normFactor;

    if (it == 0)
    {
        solverPerf.initialResidual() = residual;
    }

    if (residual < abstol)
    {
        *reason = KSP_CONVERGED_ATOL;
    }
    else if
    (
        rtol > Foam::SMALL  /** pTraits<Type>::one */
     && residual < rtol * solverPerf.initialResidual()
    )
    {
        *reason = KSP_CONVERGED_RTOL;
    }
    else if (it >= maxits)
    {
        *reason = KSP_DIVERGED_ITS;
    }
    else if (residual > divtol)
    {
        *reason = KSP_DIVERGED_DTOL;
    }
    solverPerf.finalResidual() = residual;

    PetscFunctionReturn(0);
}


void Foam::PetscUtils::setFlag
(
    const word& key,
    const word& val,
    const bool verbose
)
{
    if (verbose)
    {
        Info<< key << ' ' << val << nl;
    }

    PetscOptionsSetValue(NULL, key.c_str(), val.c_str());
}


void Foam::PetscUtils::setFlags
(
    const word& prefix,
    const dictionary& dict,
    const bool verbose
)
{
    for (const entry& e : dict)
    {
        const word key = '-' + prefix + e.keyword();
        //const word val = e.get<word>();
        word val;
        ITstream& is = e.stream();
        is >> val;

        
        if (verbose)
        {
            Info<< key << ' ' << val << nl;
        }

        PetscOptionsSetValue(NULL, key.c_str(), val.c_str());
    }
}

// ************************************************************************* //
