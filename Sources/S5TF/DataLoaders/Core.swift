import TensorFlow

// MARK: - S5TFBatch
public protocol S5TFBatch {}

public struct S5TFUnlabeledBatch<T: TensorFlowScalar>: S5TFBatch {
    public let data: Tensor<T>
}

public struct S5TFLabeledBatch<T: TensorFlowScalar>: S5TFBatch {
    public var data: Tensor<T>
    public var labels: Tensor<Int32>

    public init(data: Tensor<T>, labels: Tensor<Int32>) {
        self.data = data
        self.labels = labels
    }
}

// MARK: - S5TFDataLoader
public protocol Index {}
extension Int: Index {}
extension String: Index {}

public protocol S5TFDataLoader {
    func load() -> [Index]
    func createBatch(from indices: [Index]) -> S5TFBatch
}

// MARK: - S5TFDataIterator
public struct S5TFDataIterator: Sequence, IteratorProtocol {
    public typealias Element = S5TFBatch
    private let indices: [Index]
    private var index: Int = 0
    private var dataLoader: S5TFDataLoader

    public let batchSize: Int?

    var count: Int {
        return indices.count
    }

    public init(dataLoader: S5TFDataLoader) {
        let indices = dataLoader.load()
        self.init(
            indices: indices,
            dataLoader: dataLoader
        )
    }

    private init(
        indices: [Index],
        batchSize: Int? = nil,
        dataLoader: S5TFDataLoader
    ) {
        self.indices = indices
        self.batchSize = batchSize
        self.dataLoader = dataLoader
    }

    public func batched(_ batchSize: Int) -> S5TFDataIterator {
        guard batchSize >= 1 else {
            fatalError("Batch size must be greater than or equal to 1")
        }

        guard batchSize <= count else {
            fatalError("Batch size equal to or smaller than the number of items.")
        }

        return S5TFDataIterator(
            indices: self.indices,
            batchSize: batchSize,
            dataLoader: dataLoader
        )
    }

    public func shuffled() -> S5TFDataIterator {
        return S5TFDataIterator(
            indices: self.indices.shuffled(),
            batchSize: self.batchSize,
            dataLoader: dataLoader
        )
    }

    public mutating func next() -> S5TFBatch? {
        guard let batchSize = batchSize else {
            fatalError("This data loader does not have a batch size. Set a batch size by calling `.batched(...)`")
        }

        guard (index) <= (count - 1) else {
            return nil
        }

        // Use a partial batch is fewer items than the batch size are available.
        let thisBatchSize = Swift.min(count - index, batchSize)
        let indices = [Int](index..<index + thisBatchSize).map({ self.indices[$0] })
        self.index += thisBatchSize
        return dataLoader.createBatch(from: indices)
    }
}

// MARK: - MyDataLoader
public struct MyDataLoader: S5TFDataLoader {
    public func load() -> [Index] {
        // Download the data here. You return a list of indexes that can be used later on.
        return [0, 1, 2, 3, 4, 5, 6, 7]
    }

    public func createBatch(from indices: [Index]) -> S5TFBatch {
        // Call `loadItem(at:)`, an optional, private helper function.
        return S5TFUnlabeledBatch<Float>(data: Tensor<Float>(indices.map { loadItem(at: $0) }))
    }

    private func loadItem(at index: Index) -> Float {
        // This is where you would load an image from disk, get it from the arary, etc.
        return Float(index as! Int) * 2
    }
}
