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
public typealias Index = Int

public protocol S5TFDataLoader: Sequence, IteratorProtocol where Element: S5TFBatch {
    var indices: [Index] { get }
    var batchSize: Int? { get }

    init(
        indices: [Index],
        batchSize: Int?
    )

    func createBatch(from indicides: [Index]) -> Element
}

public extension S5TFDataLoader {
    var count: Int {
        return indices.count
    }

    func batched(_ batchSize: Int) -> Self {
        guard batchSize >= 1 else {
            fatalError("Batch size must be greater than or equal to 1")
        }

        guard batchSize <= count else {
            fatalError("Batch size equal to or smaller than the number of items.")
        }
        
        return Self.init(
            indices: self.indices,
            batchSize: batchSize
        )
    }

    func shuffled() -> Self {
        return Self.init(
            indices: self.indices.shuffled(),
            batchSize: self.batchSize
        )
    }
}
