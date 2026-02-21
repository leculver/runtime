// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Collections.Generic;

namespace Microsoft.Diagnostics.DataContractReader.Data;

internal sealed class EETypeHashTable : IData<EETypeHashTable>
{
    private const ulong FLAG_MASK = 0x1ul;

    private readonly DacEnumerableHash _baseHashTable;
    private readonly Target _target;

    static EETypeHashTable IData<EETypeHashTable>.Create(Target target, TargetPointer address) => new EETypeHashTable(target, address);
    public EETypeHashTable(Target target, TargetPointer address)
    {
        _target = target;
        Target.TypeInfo type = target.GetTypeInfo(DataType.EETypeHashTable);

        _baseHashTable = new(target, address, type);

        List<Entry> entries = [];
        foreach (TargetPointer entry in _baseHashTable.Entries)
        {
            TargetPointer typeHandle = target.ReadPointer(entry);
            entries.Add(new(typeHandle));
        }
        Entries = entries;
    }

    public IReadOnlyList<Entry> Entries { get; init; }

    /// <summary>
    /// Returns type handle entries whose stored hash value matches the given hash,
    /// using bucket-based lookup instead of scanning all entries.
    /// </summary>
    public IEnumerable<Entry> FindByHash(uint hash)
    {
        foreach (DacEnumerableHash.HashedEntry hashedEntry in _baseHashTable.FindEntriesByHash(hash))
        {
            TargetPointer typeHandle = _target.ReadPointer(hashedEntry.Value);
            yield return new Entry(typeHandle);
        }
    }

    public readonly struct Entry(TargetPointer value)
    {
        public TargetPointer TypeHandle { get; } = value & ~FLAG_MASK;
        public uint Flags { get; } = (uint)(value.Value & FLAG_MASK);
    }
}
