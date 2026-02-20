// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
//*****************************************************************************
// File: executioncontrol.cpp
//
// Implementation of execution control for interpreter breakpoints.
//
//*****************************************************************************

#include "stdafx.h"
#include "executioncontrol.h"
#include "controller.h"
#include "../../vm/codeman.h"

#ifdef FEATURE_INTERPRETER
#include "../../interpreter/intops.h"
#endif

#if !defined(DACCESS_COMPILE)
#ifdef FEATURE_INTERPRETER

//=============================================================================
// InterpreterExecutionControl - Interpreter bytecode breakpoints
//=============================================================================

InterpreterExecutionControl InterpreterExecutionControl::s_instance;

InterpreterExecutionControl* InterpreterExecutionControl::GetInstance()
{
    return &s_instance;
}

bool InterpreterExecutionControl::ApplyPatch(CORDB_ADDRESS_TYPE* address, PRD_TYPE& originalOpcode)
{
    _ASSERTE(address != NULL);

    LOG((LF_CORDB, LL_INFO10000, "InterpreterEC::ApplyPatch at bytecode addr %p\n", address));

    originalOpcode = *(int32_t*)address;
    *(uint32_t*)address = INTOP_BREAKPOINT;

    LOG((LF_CORDB, LL_EVERYTHING, "InterpreterEC::ApplyPatch Breakpoint inserted at %p, saved opcode %x\n",
        address, originalOpcode));

    return true;
}

bool InterpreterExecutionControl::UnapplyPatch(CORDB_ADDRESS_TYPE* address, PRD_TYPE originalOpcode)
{
    _ASSERTE(address != NULL);

    LOG((LF_CORDB, LL_INFO1000, "InterpreterEC::UnapplyPatch at bytecode addr %p, replacing with original opcode 0x%x\n",
        address, originalOpcode));

    *(uint32_t*)address = (uint32_t)originalOpcode;

    LOG((LF_CORDB, LL_EVERYTHING, "InterpreterEC::UnapplyPatch Restored opcode at %p\n", address));

    return true;
}

#endif // FEATURE_INTERPRETER
#endif // !DACCESS_COMPILE