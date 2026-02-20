// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
//*****************************************************************************
// File: executioncontrol.h
//
// Abstraction for breakpoint and single-step operations across different
// code execution strategies (JIT, interpreter, R2R).
// TODO: Currently only interpreter is supported. https://github.com/dotnet/runtime/issues/120842
//
//*****************************************************************************

#ifndef EXECUTIONCONTROL_H_
#define EXECUTIONCONTROL_H_

#ifndef CORDB_ADDRESS_TYPE
typedef const BYTE                  CORDB_ADDRESS_TYPE;
#endif

#ifndef PRD_TYPE
#define PRD_TYPE                    DWORD_PTR
#endif

#ifdef FEATURE_INTERPRETER


class IExecutionControl
{
public:
    virtual ~IExecutionControl() = default;

    virtual bool ApplyPatch(CORDB_ADDRESS_TYPE* address, PRD_TYPE& originalOpcode) = 0;
    virtual bool UnapplyPatch(CORDB_ADDRESS_TYPE* address, PRD_TYPE originalOpcode) = 0;
};

typedef DPTR(IExecutionControl) PTR_IExecutionControl;

// Interpreter execution control using bytecode patching
class InterpreterExecutionControl : public IExecutionControl
{
public:
    static InterpreterExecutionControl* GetInstance();

    // Apply a breakpoint patch -- saves the original opcode and writes a breakpoint
    virtual bool ApplyPatch(CORDB_ADDRESS_TYPE* address, PRD_TYPE& originalOpcode) override;

    // Remove a breakpoint patch -- restores the original opcode
    virtual bool UnapplyPatch(CORDB_ADDRESS_TYPE* address, PRD_TYPE originalOpcode) override;

private:
    InterpreterExecutionControl() = default;
    static InterpreterExecutionControl s_instance;
};

#endif // FEATURE_INTERPRETER
#endif // EXECUTIONCONTROL_H_