// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using Microsoft.Diagnostics.DataContractReader.Contracts;
using Microsoft.Diagnostics.DataContractReader.TestInfrastructure;
using Xunit;

namespace Microsoft.Diagnostics.DataContractReader.Tests;

public class GCInfoTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    [InlineData(6)]
    public void DecodeGCInfoRejectsUnsupportedVersion(uint gcVersion)
    {
        TestPlaceholderTarget target = new TestPlaceholderTarget.Builder(new MockTarget.Architecture { IsLittleEndian = true, Is64Bit = true })
            .AddGlobalStrings((Constants.Globals.Architecture, "x64"))
            .AddContract<IRuntimeInfo>(version: "c1")
            .AddContract<IGCInfo>(version: "c1")
            .Build();

        Assert.Throws<NotSupportedException>(() => target.Contracts.GCInfo.DecodePlatformSpecificGCInfo(TargetPointer.Null, gcVersion));
        Assert.Throws<NotSupportedException>(() => target.Contracts.GCInfo.DecodeInterpreterGCInfo(TargetPointer.Null, gcVersion));
    }
}
