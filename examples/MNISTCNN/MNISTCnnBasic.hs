{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE AllowAmbiguousTypes #-}

module Main where

import           Codec.Picture
import           Control.Category
import qualified Data.Array       as A
import           Data.MyPrelude
import           Data.Proxy       as DP
import           Data.Utils
import qualified Data.Vector.Storable as VS
import           Numeric.Neural
import           Pipes.GZip       (decompress)
import qualified Pipes.Prelude    as P
import           Prelude          hiding (id, (.))

main :: IO ()
main = flip evalRandT (mkStdGen 999999) $ do
    xs     <- getSamples [0 .. 999]
    m      <- modelR (whiten mnistModel $ fst <$> xs)
    liftIO $ printf "\n> Starting training\n\n"
    liftIO $ printf "generation  learning rate  batch error\n\n"
    (a, g) <- runEffect $
            cachingBatchP getSamples 60000 20 2000 100
        >-> descentP m 1 (\g -> 0.12 * 100 / (100 + fromIntegral g))  
        >-> reportTSP 1 report
        >-> consumeTSP check
    liftIO $ printf "\nreached accuracy of %f after %d generations\n" a g

  where

    getSamples xs = liftIO $ runSafeT $ P.toListM $ trainSamples >-> indicesP xs

    report ts = liftIO $ do
        let g = tsGeneration ts
        when (g `mod` 5 == 0) $ do
          printf "   %7d       %8.6f   %10.8f\n" g (tsEta ts) (tsBatchError ts)

    check ts = do
        let g = tsGeneration ts
        if g `mod` 50 == 0
            then do
                liftIO $ printf "\n> Calculating accuracy\n\n"
                a <- liftIO $ accuracy $ tsModel ts
                liftIO $ printf "accuracy %f\n\n" a
                return $ if a > 0.9 then Just (a, g) else Nothing
            else return Nothing

accuracy :: MNISTModel -> IO Double
accuracy m = runSafeT $ fromJust <$> classifierAccuracyP m testSamples

type Img = Image Pixel8

data Digit = Zero | One | Two | Three | Four | Five | Six | Seven | Eight | Nine
    deriving (Show, Read, Eq, Ord, Enum, Bounded)

type Sample = (Img, Digit)

trainImagesFile, trainLabelsFile, testImagesFile, testLabelsFile :: FilePath
trainImagesFile = "examples" </> "MNIST" </> "train-images-idx3-ubyte" <.> "gz"
trainLabelsFile = "examples" </> "MNIST" </> "train-labels-idx1-ubyte" <.> "gz"
testImagesFile  = "examples" </> "MNIST" </> "t10k-images-idx3-ubyte"  <.> "gz"
testLabelsFile  = "examples" </> "MNIST" </> "t10k-labels-idx1-ubyte"  <.> "gz"

bytes :: MonadSafe m => FilePath -> Producer Word8 m ()
bytes f = decompress (fromFile f) >-> toWord8

labels :: MonadSafe m => FilePath -> Producer Digit m ()
labels f = bytes f >-> P.drop 8 >-> P.map (toEnum . fromIntegral)

images :: MonadSafe m => FilePath -> Producer Img m ()
images f = bytes f >-> P.drop 16 >-> chunks (28 * 28) >-> P.map g

  where

    g xs = let a = A.listArray ((0, 0), (27, 27)) xs
           in  generateImage (\x y -> 255 - a A.! (y, x)) 28 28

trainSamples, testSamples :: MonadSafe m => Producer Sample m ()
trainSamples = P.zip (images trainImagesFile) (labels trainLabelsFile)
testSamples  = P.zip (images testImagesFile)  (labels testLabelsFile)

writeImg :: MonadIO m => FilePath -> Img -> m ()
writeImg f i = liftIO $ saveTiffImage (f <.> "tiff") (ImageY8 i)

type MNISTModel = Classifier (Matrix 28 28) 10 Img Digit

-- 90% accuracy after 600 generations

mnistModel :: MNISTModel
mnistModel = mkStdClassifier c i where

    c = reLULayer
      . cArr (Diff toVector)
      . conv2D
      . cArr (Diff fromMatrix)

    conv2D :: Component (Volume 28 28 1) (Volume 8 8 2)
    conv2D = convolution (Proxy :: DP.Proxy 7)  reLULayer

    i img = let m = generate $ \(x, y) -> fromIntegral (pixelAt img x y) in force m