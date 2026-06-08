# Contract GCHandle

This contract allows decoding and reading a single GC handle value. Handle table enumeration is currently provided by the GC contract rather than this standalone GCHandle contract.

## Data structures defined by contract
``` csharp
struct DacGCHandle
{
    DacGCHandle(TargetPointer value) { Value = value; }
    TargetPointer Value;
}
```

## Apis of contract
``` csharp
TargetPointer GetObject(DacGCHandle gcHandle);
```

## Version 1

``` csharp
TargetPointer GetObject(DacGCHandle gcHandle)
{
    if (gcHandle.Value == TargetPointer.Null)
        return TargetPointer.Null;
    return Target.ReadTargetPointer(gcHandle.Value);
}
```
